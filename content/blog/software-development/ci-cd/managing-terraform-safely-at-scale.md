---
title: "Managing Terraform safely at scale: Plan, policy, and blast radius"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how to operate Terraform safely in a team environment: plan-in-CI, policy-as-code with OPA and Sentinel, layered state for blast-radius control, and drift detection that catches infrastructure rot before it causes incidents."
tags:
  [
    "ci-cd",
    "devops",
    "terraform",
    "infrastructure-as-code",
    "policy-as-code",
    "opa",
    "state-management",
    "platform-engineering",
    "cloud-infrastructure",
    "gitops",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/managing-terraform-safely-at-scale-1.png"
---

There is a specific kind of dread that only Terraform operators know. You are twelve minutes into a `terraform apply` on a Friday afternoon. The plan looked fine — sixty-something resources changing, all `~` (in-place updates), nothing alarming. Then the terminal pauses. Then it starts printing red lines. `aws_db_instance.production: Destroying...`. Your stomach drops. You didn't see that. You didn't plan for that. Your production database, the one holding three years of customer data, is being deleted by a CI bot that has full AWS credentials and no one to ask for permission.

That scenario is not hypothetical. It happens to teams who have automated `terraform apply` without putting any safety structure around it. The scary part is not that Terraform made a mistake — Terraform did exactly what you told it to do. The scary part is that the team had no layer of defense between "code merged to main" and "apply runs on prod." No plan review. No policy check. No approval gate. No blast-radius limit. Just `terraform apply -auto-approve` in a GitHub Actions job, running with an IAM role that can do everything.

This post is about building all those layers of defense. By the end you will know how to run `terraform plan` automatically on every PR and post the output as a PR comment, how to wire an OPA/Conftest policy check that blocks public S3 buckets before the apply ever runs, how to split a monolith state into layers that bound your blast radius by a factor of ten, how to read a plan output so that forced replacements never surprise you, and how to detect infrastructure drift before it accumulates into the incident that woke you up at 3 AM. Every concept is grounded in runnable GitHub Actions YAML, real HCL, and real Rego policy. This is Track D3 of the [CI/CD & Cloud-Native Delivery series](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the infrastructure-as-code delivery layer.

![Plan-in-CI flow showing a PR triggering an automated Terraform plan that branches based on whether destroys are detected, routing clean plans through auto-approval and destructive plans through a manual gate before apply](/imgs/blogs/managing-terraform-safely-at-scale-1.png)

## 1. Why Terraform safety is a delivery problem

Before diving into tools, it helps to ground this in the CI/CD mental model. The four DORA metrics — deploy frequency, lead time for changes, change-failure rate, and time-to-restore — apply to infrastructure changes exactly as they apply to application deployments. When your infrastructure change-failure rate is high (say, 30% of Terraform applies require a manual remediation), every engineer learns to fear applies and stops shipping infrastructure changes frequently. Lead time balloons because teams batch changes to reduce the number of applies, which makes each apply bigger, which increases risk further. This is the same doom loop as infrequent, large application deployments.

The fix is the same. Small, frequent, tested, reviewed infrastructure changes. The tooling to achieve this safely is what this post covers.

The commit → build → test → package → deploy → operate spine that governs the application delivery pipeline has a direct analogue for infrastructure:

- **Commit**: engineer writes HCL, opens a PR.
- **Plan** (the "test"): `terraform plan` runs automatically, producing a human-readable diff of what would change.
- **Policy check** (the "gate"): OPA/Conftest or Sentinel validates the plan JSON against policy rules.
- **Review + approve**: a human (or a no-destroys-detected auto-approve rule) signs off on the plan output.
- **Apply** (the "deploy"): CI runs `terraform apply` with the exact saved plan file from step 2 — not a fresh plan.
- **Drift detection** (the "operate"): a scheduled job runs `terraform plan` nightly and alerts on any unexpected differences.

The "build once, promote everywhere" principle has an IaC equivalent: **plan once, apply exactly that plan.** If you run `terraform plan` for review, then run a fresh `terraform apply` five minutes later, you have broken the chain of trust. Something might have changed in the five minutes between plan and apply. Use `-out` to save the plan file as an artifact; use that artifact in the apply step.

### The IaC delivery maturity ladder

Teams implementing Terraform safety typically move through four levels of maturity, each building on the last:

**Level 0 — Local, unguarded applies.** Engineers run `terraform apply` from their laptops with personal IAM credentials. There is no shared state, or the shared state is accessed inconsistently. There are no plan reviews. Applies happen "when someone remembers." Change-failure rate: typically 25–40% for Terraform-related changes. MTTR when something goes wrong: measured in hours because nobody has an audit trail of what changed.

**Level 1 — Remote state + plan before apply (manual).** The team has a shared S3 backend with state locking. Engineers are expected to run `terraform plan` locally before applying. Applies still happen locally. Change-failure rate drops to 15–20% because at least someone is looking at the plan before applying. The problem: consistency is enforced by convention, not by tooling. Tired engineers skip the plan review. Two engineers on different versions of Terraform get different plan outputs.

**Level 2 — Plan-in-CI with PR comment and manual approval gate.** Every PR triggers an automated `terraform plan` on a standardized CI runner. The plan output is posted as a PR comment. Merges require a human to have read the plan. Applies run in CI after merge, not locally. Change-failure rate: typically 5–8%. Plan reviews become consistent because the plan is right there in the PR, not a separate manual step.

**Level 3 — Policy-as-code, layered state, drift detection.** OPA/Conftest policies run automatically against every plan JSON. State is split into layers by ownership and blast radius. Nightly drift detection alerts on any deviation. The CI bot uses OIDC-federated scoped roles, not long-lived keys. Apply-in-CI is enforced by IAM policy (the apply role is not assumable locally). Change-failure rate: typically 2–4%. Drift is caught within 24 hours instead of being discovered during an incident.

Most teams land at Level 1 by accident and stay there for years. The goal of this post is to help you reach Level 3 deliberately, one piece at a time, without burning a sprint re-architecting everything at once.

### Why infrastructure changes fail more than application changes

A useful observation from the DORA research: teams with high software delivery performance also tend to have more reliable infrastructure change processes. The causal chain runs in both directions. Better delivery practices reduce change-failure rate, and lower change-failure rate gives teams the confidence to deliver more frequently.

Infrastructure changes have some specific failure modes that application changes do not:

**State file drift.** Application deployments are stateless from the delivery system's perspective — the artifact is deployed, the old version is stopped. Terraform carries state. If the state file diverges from reality (someone made a manual change), the next plan may be wrong in ways that are hard to predict.

**Resource replacement semantics.** Changing some Terraform arguments requires destroying and recreating the resource. Changing a container image tag does not destroy a running container — it triggers a rolling update. But changing the `engine_version` on an RDS instance may require a destroy-and-recreate. This asymmetry surprises engineers who are used to application deployments where changes are always in-place.

**Cross-resource dependency chains.** Changing a security group ID that is referenced by 15 other resources triggers a cascade of changes in the plan. Application deployments are typically one service at a time. Terraform plans can ripple across dozens of resources in ways that are hard to anticipate without reading the full plan output.

**Slow feedback loops.** A container build fails in 30 seconds. A `terraform plan` on a large state takes 10 minutes. Engineers working in a slow feedback loop make more mistakes per change because they are more likely to multitask, context-switch, and lose focus on the details.

All four failure modes are addressed by the practices in this post. State file drift is addressed by drift detection. Resource replacement semantics are surfaced by reading the plan carefully (the `-/+` symbol). Cross-resource dependency chains are visible in the plan output and bounded by state layering. Slow feedback loops are solved by layering state (each layer plans in under 90 seconds instead of 10 minutes).

## 2. Plan-in-CI: the first line of defense

The most impactful single change you can make to a Terraform workflow is this: every pull request triggers an automatic `terraform plan`, and the output is posted as a comment on the PR before any human approves the merge.

This sounds obvious. Most teams do not do it. They rely on engineers running `terraform plan` locally before pushing. The problem with local plans is that they depend on the engineer's local credentials, local Terraform version, and local workspace state. Two engineers on the same team may get different plan outputs for the same commit. A plan on a CI runner with a standardized environment and the real remote state is the only plan you can trust.

Local plans also create an asymmetric visibility problem. The person who ran the plan (the PR author) saw the output. The reviewers (who are being asked to approve and merge the change) have not. When a reviewer approves a PR, they are trusting that the author ran a plan, read it carefully, and would have noticed anything alarming. That trust is often misplaced — not because engineers are careless, but because the cognitive load of running a local plan, reading it, and deciding it is safe is high, and humans under time pressure cut corners.

Plan-in-CI solves both problems simultaneously. The plan runs on a standardized CI environment, producing a consistent, reproducible output. The output appears in the PR where every reviewer can read it, creating shared visibility. The reviewer can see exactly what will change without trusting the author's recall. If the plan shows `3 to destroy`, every reviewer sees those three destroys and can decide whether they look intentional.

Here is the principle in arithmetic. If you apply Terraform N times per month and each apply has a P probability of containing a silent destructive change that a plan review would catch, your expected number of silent destructions per month is N × P. If N = 30 (daily applies) and P = 0.05 (one in twenty applies has a hidden destroy), you expect 1.5 incidents per month from this cause alone. Enforcing plan-in-CI with a human review step reduces P toward zero. Even if you reduce P from 0.05 to 0.01, you go from 1.5 incidents to 0.3 — a 5x improvement in change-failure rate for this category alone.

### Terraform init and the backend configuration

Before `terraform plan` can run in CI, `terraform init` must download the providers and configure the backend. This is an often-overlooked detail in CI setups. The backend configuration (which S3 bucket holds the state, which DynamoDB table handles locking) should not be hardcoded in the `backend.tf` file, because different environments use different state buckets. Instead, use partial backend configuration and pass the environment-specific values at init time:

```hcl
# infra/app/backend.tf
terraform {
  backend "s3" {
    # Populated at init time via -backend-config flags
    # bucket, key, region are passed in CI
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}
```

```bash
# In CI, run init with backend configuration:
terraform init \
  -backend-config="bucket=my-terraform-state-prod" \
  -backend-config="key=app/terraform.tfstate" \
  -backend-config="region=us-east-1"
```

This pattern lets the same Terraform code target different environment backends without modifying the source code. The S3 bucket name and state key are environment-specific; they come from CI variables, not from committed code.

State locking via DynamoDB is critical when multiple CI jobs might plan or apply concurrently. Without locking, two simultaneous applies on the same state file produce corrupted state. DynamoDB provides atomic lock acquisition — whichever `terraform init` gets the lock first proceeds; the second one waits or fails with a clear error message. Always configure this; the cost is negligible (a few pennies per month in DynamoDB read/write units).

#### Worked example: GitHub Actions plan-in-CI with PR comment

Here is a complete GitHub Actions workflow that runs `terraform plan` on every PR, saves the plan as an artifact, and posts the output as a PR comment. This is the foundation everything else builds on.

```yaml
# .github/workflows/terraform-plan.yml
name: Terraform Plan

on:
  pull_request:
    branches: [main]
    paths:
      - "infra/**"

permissions:
  contents: read
  pull-requests: write
  id-token: write # OIDC federation to AWS

env:
  TF_VERSION: "1.8.5"
  AWS_REGION: "us-east-1"

jobs:
  plan:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: infra/app

    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-ci-plan-role
          aws-region: ${{ env.AWS_REGION }}

      - name: Terraform Init
        run: terraform init -backend-config="bucket=my-terraform-state" -backend-config="key=app/terraform.tfstate"

      - name: Terraform Format Check
        run: terraform fmt -check -recursive

      - name: Terraform Validate
        run: terraform validate

      - name: Terraform Plan
        id: plan
        run: |
          terraform plan \
            -out=tfplan.binary \
            -var-file=environments/prod.tfvars \
            2>&1 | tee plan_output.txt
          echo "exit_code=$?" >> "$GITHUB_OUTPUT"
        continue-on-error: true

      - name: Convert plan to JSON (for policy checks)
        run: terraform show -json tfplan.binary > tfplan.json

      - name: Upload plan artifacts
        uses: actions/upload-artifact@v4
        with:
          name: tfplan-${{ github.sha }}
          path: |
            infra/app/tfplan.binary
            infra/app/tfplan.json
          retention-days: 7

      - name: Post plan comment on PR
        uses: actions/github-script@v7
        if: github.event_name == 'pull_request'
        with:
          script: |
            const fs = require('fs');
            const planOutput = fs.readFileSync('infra/app/plan_output.txt', 'utf8');
            const truncated = planOutput.length > 65000
              ? planOutput.substring(0, 65000) + '\n...(truncated)'
              : planOutput;

            const body = `## Terraform Plan

            <details><summary>Show Plan</summary>

            \`\`\`
            ${truncated}
            \`\`\`

            </details>

            **Plan exit code**: \`${{ steps.plan.outputs.exit_code }}\`
            **Commit**: \`${{ github.sha }}\``;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

      - name: Fail if plan errored
        if: steps.plan.outputs.exit_code != '0' && steps.plan.outputs.exit_code != '2'
        run: exit 1
```

The exit code semantics matter. Terraform returns exit code 0 when no changes are planned, 2 when changes are planned, and 1 on error. The step uses `continue-on-error: true` so it can capture the output even on planned changes, then explicitly fails on exit code 1 (genuine errors).

The OIDC configuration is critical. The CI runner assumes an IAM role via OpenID Connect federation — no long-lived AWS access keys in GitHub Secrets. The role should have read-only permissions for the plan step: `sts:AssumeRole` is the only secret the CI runner needs.

## 3. The apply gate: plan → approve → apply

A plan that nobody reads is useless. The workflow above posts the plan to the PR, but humans need to actually read it and approve the merge before the apply runs. The apply job is a separate workflow, triggered on push to `main` (after merge), and it uses the exact plan artifact from the plan step — not a freshly generated plan.

The reasoning for using the saved plan artifact (not a fresh plan in the apply job) comes from the "plan once, apply exactly that plan" principle. Consider what can happen between when a PR is merged and when the apply job starts:

- Another engineer's PR merged ten seconds earlier (a race condition) and their apply is running right now. Your fresh plan will see that engineer's changes in the state as completed, and may produce a different output than the plan your reviewer approved.
- A provider API is experiencing intermittent failures. A fresh plan might fail, producing an error that blocks the apply. The saved plan binary is self-contained and does not re-query providers.
- A data source lookup (e.g., `data.aws_ami.latest`) returns a different AMI ID than it did during the plan step. This is exactly the kind of unreviewed change that causes incidents.

The saved plan binary is Terraform's cryptographic commitment to a specific set of changes. It contains the expected state of every resource before and after the apply. When you run `terraform apply tfplan.binary`, Terraform verifies the current state matches what the plan expected before proceeding. If something changed between plan and apply, Terraform errors rather than applying a plan that no longer matches reality.

### Handling plan artifact expiry

Plan artifacts have a practical expiry. If your PR sits unmerged for a week, the plan binary is stale. The state may have changed substantially since the plan was generated. Running a stale plan can produce incorrect results.

Implement a maximum age check in your apply workflow:

```yaml
- name: Check plan artifact age
  run: |
    # GitHub Actions artifact metadata stores creation time
    # If the PR was opened more than 48 hours ago, re-plan
    PR_CREATED_AT="${{ github.event.pull_request.created_at }}"
    CREATED_EPOCH=$(date -d "$PR_CREATED_AT" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$PR_CREATED_AT" +%s)
    NOW_EPOCH=$(date +%s)
    AGE_HOURS=$(( (NOW_EPOCH - CREATED_EPOCH) / 3600 ))
    if [ "$AGE_HOURS" -gt 48 ]; then
      echo "Plan artifact is ${AGE_HOURS} hours old — re-plan required before apply"
      exit 1
    fi
```

For long-lived PRs, require the engineer to push a new commit (or use a `/replan` comment command in Atlantis) to generate a fresh plan before the merge is permitted.

### The two-role OIDC model

The plan workflow and the apply workflow should use different IAM roles with different permission scopes. This is the smallest-possible-privilege design for Terraform CI:

| Role | Permissions | Assumable by |
|------|------------|--------------|
| `terraform-ci-plan-role` | Read-only: `Describe*`, `List*`, `Get*` on all services; `s3:GetObject` on state bucket; `dynamodb:GetItem` on lock table | Plan job on any branch of the repo |
| `terraform-ci-apply-role` | Write permissions for the exact resources managed by this state layer | Apply job on the main branch only |

The plan role can be assumed broadly — from any branch, including feature branches and forks. This lets engineers get plan feedback early without security risk. A read-only plan role cannot destroy anything.

The apply role is restricted by the OIDC condition to the main branch only. A malicious PR cannot contain a GitHub Actions workflow that calls `terraform apply` with the apply role — the OIDC subject claim will be `repo:myorg/myrepo:ref:refs/pull/123/head` (a PR branch), not `repo:myorg/myrepo:ref:refs/heads/main`, and the IAM policy will reject the role assumption.

```yaml
# .github/workflows/terraform-apply.yml
name: Terraform Apply

on:
  push:
    branches: [main]
    paths:
      - "infra/**"

permissions:
  contents: read
  id-token: write

jobs:
  apply:
    runs-on: ubuntu-latest
    environment: production # GitHub Environments approval gate
    defaults:
      run:
        working-directory: infra/app

    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.8.5"

      - name: Configure AWS credentials (OIDC, apply role)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-ci-apply-role
          aws-region: us-east-1

      - name: Download plan artifact
        uses: actions/download-artifact@v4
        with:
          name: tfplan-${{ github.sha }}
          path: infra/app/

      - name: Terraform Init
        run: terraform init -backend-config="bucket=my-terraform-state" -backend-config="key=app/terraform.tfstate"

      - name: Terraform Apply (saved plan)
        run: terraform apply -auto-approve tfplan.binary
```

The `environment: production` line is load-bearing. GitHub Environments let you configure required reviewers — specific people or teams who must approve before the job runs. Without this, the apply runs automatically on every push to main, which defeats the purpose of the plan review.

Two separate IAM roles are used: a read-only plan role (can call `ec2:Describe*`, `s3:GetObject` on the state bucket, etc.) and a write-capable apply role (can create/modify/delete resources). The apply role should only be assumable by the apply workflow — not by engineers' local credentials, not by the plan workflow. If the apply role leaks, you want the blast radius of that leak to be limited to the CI system, not to any engineer's laptop.

## 4. Policy-as-code: OPA, Conftest, and Sentinel

The plan review gate relies on a human reading the plan output carefully. Humans are slow, inconsistent, and tired at 5 PM on a Friday. Policy-as-code automates the consistency layer: define rules in code, evaluate them against every plan, fail the CI if any rule is violated.

![Policy-as-code enforcement stack showing Terraform plan JSON flowing through OPA/Conftest policy check then Sentinel policies to produce a signed approved plan artifact before apply](/imgs/blogs/managing-terraform-safely-at-scale-4.png)

There are two main policy-as-code tools in the Terraform ecosystem:

**OPA/Conftest** is an open-source, language-agnostic policy engine. You write policies in Rego (a declarative query language), run Conftest against the plan JSON, and it exits non-zero if any policy is violated. Conftest integrates naturally into any CI system as a CLI command.

**HashiCorp Sentinel** is a policy-as-code framework built into HCP Terraform (formerly Terraform Cloud) and Terraform Enterprise. It has tighter integration with the Terraform run lifecycle but requires HCP Terraform. If you are on open-source Terraform with a self-hosted runner, OPA/Conftest is the practical choice.

Here is an OPA policy that blocks public S3 buckets:

```rego
# policies/deny_public_s3.rego
package terraform.policies

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Deny any S3 bucket with a public ACL
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_s3_bucket"
    resource.change.after.acl in ["public-read", "public-read-write", "authenticated-read"]
    msg := sprintf(
        "S3 bucket '%s' has a public ACL '%s'. Public buckets are not allowed.",
        [resource.address, resource.change.after.acl]
    )
}

# Deny any S3 bucket ACL resource with a public ACL (newer resource split)
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_s3_bucket_acl"
    resource.change.after.acl in ["public-read", "public-read-write"]
    msg := sprintf(
        "S3 bucket ACL resource '%s' sets a public ACL '%s'. Public ACLs are not allowed.",
        [resource.address, resource.change.after.acl]
    )
}

# Deny any S3 bucket with public access block disabled
deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_s3_bucket_public_access_block"
    resource.change.after.block_public_acls == false
    msg := sprintf(
        "S3 bucket public access block '%s' has block_public_acls=false. All buckets must block public access.",
        [resource.address]
    )
}
```

And a policy that enforces required tags on all resources:

```rego
# policies/require_tags.rego
package terraform.policies

import future.keywords.contains
import future.keywords.if
import future.keywords.in

required_tags := {"Environment", "Owner", "CostCenter", "Team"}

taggable_types := {
    "aws_instance",
    "aws_s3_bucket",
    "aws_rds_cluster",
    "aws_elasticache_replication_group",
    "aws_eks_cluster",
    "aws_vpc",
    "aws_security_group",
}

deny contains msg if {
    some resource in input.resource_changes
    resource.type in taggable_types
    resource.change.actions != ["delete"]
    some required_tag in required_tags
    not resource.change.after.tags[required_tag]
    msg := sprintf(
        "Resource '%s' (type %s) is missing required tag '%s'.",
        [resource.address, resource.type, required_tag]
    )
}
```

And one that restricts EC2 instance types to an approved list:

```rego
# policies/allowed_instance_types.rego
package terraform.policies

import future.keywords.contains
import future.keywords.if
import future.keywords.in

allowed_types := {
    "t3.small", "t3.medium", "t3.large",
    "m6i.large", "m6i.xlarge", "m6i.2xlarge",
    "c6i.large", "c6i.xlarge",
    "r6i.large",
}

deny contains msg if {
    some resource in input.resource_changes
    resource.type == "aws_instance"
    resource.change.actions != ["delete"]
    not resource.change.after.instance_type in allowed_types
    msg := sprintf(
        "EC2 instance '%s' uses instance type '%s' which is not in the approved list: %v",
        [resource.address, resource.change.after.instance_type, allowed_types]
    )
}
```

To run Conftest in CI, add a step after the plan-to-JSON conversion:

```yaml
- name: Run OPA policies with Conftest
  uses: docker://openpolicyagent/conftest:latest
  with:
    args: test infra/app/tfplan.json --policy policies/ --namespace terraform.policies
```

Or as a shell step if you have Conftest installed on the runner:

```bash
conftest test tfplan.json --policy policies/ --namespace terraform.policies --output table
```

Conftest exits non-zero if any deny rule fires, which fails the CI job and blocks the merge. The engineer sees the policy violation in the PR, fixes the Terraform code, pushes again, and the plan and policy checks re-run.

### Sentinel on HCP Terraform

If you use HCP Terraform (formerly Terraform Cloud), Sentinel policies are configured in the workspace settings and run automatically as part of the managed run workflow. A Sentinel policy blocking public S3 buckets looks like this:

```python
# sentinel/deny_public_s3.sentinel
import "tfplan/v2" as tfplan

s3_buckets = filter tfplan.resource_changes as _, rc {
    rc.type is "aws_s3_bucket" and
    (rc.change.actions contains "create" or rc.change.actions contains "update")
}

deny_public_s3 = rule {
    all s3_buckets as _, bucket {
        bucket.change.after.acl not in ["public-read", "public-read-write", "authenticated-read"]
    }
}

main = rule {
    deny_public_s3
}
```

Sentinel policies in HCP Terraform can be set to `advisory` (warn but allow), `soft-mandatory` (block but allow override by a privileged user), or `hard-mandatory` (block unconditionally). Start with `soft-mandatory` during rollout so you can identify false positives without breaking everyone's workflows.

## 5. Understanding blast radius and why layered state matters

The concept of blast radius in infrastructure management is simple: if an apply goes wrong (wrong target, wrong variable, corrupted plan), how many resources can it touch? If the answer is "all of them," your blast radius is your entire infrastructure. That is a catastrophic risk profile.

The fear that haunts every Terraform operator is specific: "I am about to run this apply, and I do not know if it will destroy something I care about." That fear is rational when the state contains hundreds of resources. It becomes manageable when the state contains 30 resources that you can enumerate from memory.

### Why one giant state is dangerous

A team that has been using Terraform for two years without intentional state design typically ends up with a monolith state that grew organically. It started with a handful of resources and absorbed everything as the team learned Terraform. It now contains:

- VPCs, subnets, route tables, internet gateways
- Security groups for every application layer
- The production RDS cluster and its parameter groups
- The ElastiCache replication group
- The EKS cluster and all its node groups
- Every application's ALB and target groups
- IAM roles for every service
- S3 buckets, KMS keys, CloudFront distributions
- CloudWatch alarms and SNS topics

This is a common situation, not an unusual one. And it creates a specific pathology: every apply, no matter how small the change, runs a plan that re-evaluates all of these resources. Engineers become anxious before applies because they know that a single wrong attribute in any module could cascade through the dependency graph and touch something critical.

There is also a practical performance problem. Terraform refreshes every resource in the state file during a plan by making an API call to the cloud provider. With 200 resources, that is 200 API calls, some of them slow (RDS, EKS, and ElastiCache describe operations can take 2–5 seconds each). A 200-resource plan routinely takes 10–15 minutes. That is 10–15 minutes of waiting for feedback on a one-line change.

The operational consequence: engineers batch their changes to reduce the number of applies, because each apply is expensive (long feedback loop) and risky (high blast radius). Batching increases change size, which increases risk further. The death spiral continues.

### The layered state architecture

The layered state architecture solves both problems — blast radius and plan speed — by breaking the monolith into independent slices along natural ownership boundaries.

![Comparison showing a monolith state file with 200 resources and high blast radius versus a three-layer state approach with foundation, network, and app layers each containing 20-40 resources with isolated blast radius](/imgs/blogs/managing-terraform-safely-at-scale-2.png)

The blast radius of a Terraform apply is a direct function of the number of resources in the state file. If your state contains 200 resources across VPCs, subnets, EC2 instances, RDS clusters, S3 buckets, IAM roles, and EKS node groups, then every apply on that state file theoretically has access to all 200 resources. A misconfigured `target` argument, a bug in a module, or a provider returning unexpected data can cause changes to resources you never intended to touch.

The solution is layering. Split your infrastructure into independent state files, each managing a coherent slice of infrastructure:

```bash
infra/
  foundation/           # IAM roles, KMS keys, billing alarms, account-level settings
    main.tf
    backend.tf         # state bucket: s3://tf-state/foundation/terraform.tfstate
    outputs.tf         # exports: kms_key_arn, log_bucket_arn

  networking/          # VPCs, subnets, route tables, security groups, VPN, Direct Connect
    main.tf
    backend.tf         # state bucket: s3://tf-state/networking/terraform.tfstate
    data.tf            # reads foundation outputs via remote_state data source
    outputs.tf         # exports: vpc_id, private_subnet_ids, sg_ids

  platform/            # EKS cluster, RDS, ElastiCache, ECR registries
    main.tf
    backend.tf         # state bucket: s3://tf-state/platform/terraform.tfstate
    data.tf            # reads foundation + networking outputs

  app/                 # application-specific resources: ALBs, task definitions, Lambda functions
    main.tf
    backend.tf         # state bucket: s3://tf-state/app/terraform.tfstate
    data.tf            # reads platform outputs
```

The dependency graph flows downward: `app` depends on `platform`, `platform` depends on `networking`, `networking` depends on `foundation`. Each layer reads outputs from the layer below using a `terraform_remote_state` data source:

```hcl
# infra/networking/data.tf
data "terraform_remote_state" "foundation" {
  backend = "s3"
  config = {
    bucket = "my-terraform-state"
    key    = "foundation/terraform.tfstate"
    region = "us-east-1"
  }
}

# Use: data.terraform_remote_state.foundation.outputs.kms_key_arn
```

This structure has three concrete benefits beyond blast-radius reduction:

**Plan speed.** A state with 30 resources plans in roughly 30–60 seconds. A state with 200 resources plans in 8–12 minutes because Terraform makes an API call to refresh every resource's actual state. Your PR feedback loop shrinks from 12 minutes to 90 seconds.

**Independent apply cadence.** The `app` layer changes many times per day (new Lambda versions, updated ALB rules, new environment variables). The `foundation` layer changes once a quarter. Without layering, every app change re-plans the entire infrastructure, slowing down developers and exposing stable infrastructure to unnecessary risk.

**Clearer ownership.** Security engineers own `foundation`. The networking team owns `networking`. Each team can apply their layer independently without requiring coordination with every other team.

#### Worked example: blast-radius arithmetic

Before layering: 1 state file, 247 resources. An erroneous `terraform apply` during a routine app deployment has a blast radius of 247. If the team runs 20 applies per month and each has a 1% probability of touching an unintended resource due to misconfiguration, expected unintended resource touches per month = 20 × 0.01 × 247 = 49.4. That is 49 resource contaminations per month from this single risk factor.

After layering into four state files averaging 62 resources each: same 20 applies per month, same 1% misconfiguration probability, but now expected contaminations = 20 × 0.01 × 62 = 12.4. Same blast-radius reduction principle as blue-green canary deployments — limit what one action can reach.

Measured result from a real team migration (illustrative, order-of-magnitude): plan time dropped from an average of 11 minutes to 95 seconds, the change-failure rate for Terraform applies dropped from 18% to 4% over the following quarter, and the number of engineers willing to make infrastructure changes (as measured by PR authors on the IaC repo) increased by 3× because the feedback loop was no longer painful.

## 6. Reading a Terraform plan: the checklist

Most Terraform incidents are preventable by reading the plan output. The plan is Terraform's promise of what it will do. Your job, as a reviewer, is to verify that the promise matches your intent.

The plan output has three sections. First, the resource changes — one block per resource that will be created, updated, or destroyed. Second, the change summary at the bottom: `Plan: X to add, Y to change, Z to destroy.` Third, any warnings or notes about resource behaviors.

Most engineers read only the summary line. This is how the forced-replacement incident happens. The summary line says "1 to destroy, 1 to add" — which might sound like a replacement. But it might also be destroying your production RDS instance. The summary line does not tell you what is being destroyed. You have to read the resource blocks.

### The anatomy of a plan block

Here is an annotated example of a plan block showing an in-place update (safe) and a forced replacement (dangerous):

```bash
# In-place update (safe — just a tag change)
  # aws_security_group.app will be updated in-place
  ~ resource "aws_security_group" "app" {
        id                     = "sg-0a1b2c3d4e5f"
      ~ tags                   = {
          ~ "LastModified" = "2026-03-15" -> "2026-06-22"
            # (3 unchanged elements hidden)
        }
        # (8 unchanged attributes hidden)
    }

# Forced replacement (dangerous — DB will be destroyed)
  # aws_db_instance.production must be replaced
-/+ resource "aws_db_instance" "production" {
      ~ db_parameter_group_name  = "default.mysql8.0" -> "custom-prod" # forces replacement
      ~ id                       = "production-mysql-8" -> (known after apply)
      ~ identifier               = "production-mysql-8" -> (known after apply)
        # (51 unchanged attributes hidden)
    }

Plan: 1 to add, 1 to change, 1 to destroy.
```

The key reading skills:

The `~` prefix on a resource block header means in-place update. The attributes inside with `~` prefix are the ones changing. Attributes with `+` inside are being added; attributes with `-` inside are being removed. Attributes with no prefix are unchanged (and hidden by default — use `terraform show tfplan.binary` for the full output).

The `-/+` prefix on a resource block header is the signal to stop and read carefully. This resource must be destroyed and recreated. The comment `# forces replacement` appears next to the specific attribute that is causing the replacement. That is your answer to "why does this need to be replaced?" In the example above, changing `db_parameter_group_name` on an RDS instance requires recreation. That is a 4-hour database outage.

The `id` attribute changing to `(known after apply)` on a forced replacement is a secondary signal. The current resource's ID is known; the new resource's ID is not yet assigned.

### What a safe plan looks like

A plan is safe to apply when:
- The resource counts in the summary match what you expected from the change (added one security group rule → `1 to add`)
- All `-/+` replacements are for non-stateful resources (Lambda functions, IAM policies, CloudWatch alarms are generally safe to replace; RDS, ElastiCache, EC2 instances, EKS node groups are not)
- The `-` (destroy) symbols are for resources you deliberately removed from the code
- Data source refreshes are not triggering downstream replacements
- The total number of changes is in the range you expected

A plan that shows more changes than you expected is a plan to investigate, not a plan to approve. "I only changed one line, but it shows 15 resources changing" is a signal that your change touches a shared module or a variable that is referenced more broadly than you realized.

The plan uses three symbols:

- `+` (create): a new resource will be created. Generally safe.
- `-` (destroy): an existing resource will be deleted. Always read carefully.
- `~` (update in-place): an attribute will be changed without recreating the resource. Usually safe.
- `-/+` (destroy and recreate): the resource must be destroyed and recreated. This is a forced replacement. **This is the most dangerous symbol.** It means downtime for stateful resources (databases, cache clusters, EC2 instances in ASGs without proper lifecycle management).

Here is the plan review checklist:

```bash
# 1. Count the destroys and replacements — scan for these patterns first
grep -E "^  # .* will be (destroyed|replaced)" plan_output.txt

# 2. Look specifically for forced replacements
grep -E "^\s+# .* must be replaced" plan_output.txt

# 3. Count resources by action type
grep -E "Plan: [0-9]+ to add, [0-9]+ to change, [0-9]+ to destroy" plan_output.txt

# 4. Check for data source refreshes that indicate unexpected state drift
grep -E "data\." plan_output.txt | grep "will be read"
```

The specific patterns to watch for:

**Forced replacements on stateful resources.** If you see `-/+` on an `aws_rds_instance`, `aws_elasticache_replication_group`, or `aws_eks_node_group`, stop. Understand exactly why the replacement is forced (which attribute changed that requires recreation), and determine if there is a way to avoid it. Common causes: changing the `db_subnet_group_name`, changing the `engine_version` to a version that requires a major upgrade, changing `availability_zone` on a single-AZ instance.

**Data source refreshes returning unexpected values.** If a `data.aws_ami.latest` lookup is returning a different AMI ID than last week, your EC2 instances may be replaced. Pin the AMI ID in your `tfvars` files for production; use data source lookups only in dev/staging.

**Large destruction counts on re-tagged resources.** If you are adding a new tag to a module that creates 50 resources, the plan should show 50 `~` updates. If it shows 50 `-` then 50 `+`, you have a forced replacement (perhaps the resource type does not support tag updates in-place). That is 50 deletions that will cause an outage.

**Resources with `(known after apply)` on critical fields.** If the IP address of your load balancer is `(known after apply)`, your plan is telling you it cannot predict it. For most resources this is fine. For resources that other Terraform layers or applications depend on by value (not by reference), this can cause cascading failures.

### The plan review table

| Symbol | Action | Risk Level | What to verify |
|--------|--------|------------|----------------|
| `+` | Create | Low | Correct resource type, correct naming convention |
| `~` | Update in-place | Low-medium | Check if any attribute triggers replacement |
| `-/+` | Destroy + recreate | High | Is downtime acceptable? Can lifecycle rules help? |
| `-` | Destroy | Very high | Is this intentional? Does anything depend on it? |
| `<= ` | Data source read | Low | Will the new value cause a downstream replacement? |

## 7. Applying safely: -target, -refresh-only, and the apply-in-CI rule

Once the plan passes review and policy checks, the apply should run in CI — not locally. This rule has teeth.

### The state lock: your first line of defense during apply

Terraform uses state locking to prevent concurrent applies. When an apply starts, Terraform writes a lock entry to the DynamoDB table. If another apply (or plan) tries to start at the same time, it sees the lock and either waits or fails immediately, depending on the `-lock-timeout` setting.

This matters at scale because CI systems can run multiple jobs concurrently. If two PRs merge at the same time and two apply jobs start at the same time, you need the locking to prevent them from corrupting the state file. The second apply will see the lock and wait for the first to complete.

The default `-lock-timeout` is 0s (fail immediately on lock contention). For CI pipelines, setting it to a short timeout (e.g., 10 minutes) allows the second job to wait for the first to complete rather than failing:

```yaml
- name: Terraform Apply
  run: terraform apply -lock-timeout=10m -auto-approve tfplan.binary
```

If a lock is orphaned (the previous apply crashed without releasing the lock), you will see `Error acquiring the state lock`. To manually release an orphaned lock, use `terraform force-unlock <lock-id>`. This is an operator action, not something CI should do automatically.

### Handling state lock timeouts gracefully

A common CI failure mode: an apply job starts, acquires the state lock, and then fails partway through due to a provider API error or network timeout. Terraform has partially applied some resources and partially not. The state file reflects the completed changes, but not the incomplete ones. The next plan will show the remaining work.

This partial-apply situation is normal and recoverable. Terraform is designed to handle it: just run another plan-and-apply cycle. The next apply will pick up where the first left off, because the state file accurately records which resources were successfully created or modified.

The danger is engineers who see a partial apply and panic. They manually modify resources in the console to "fix" things, which creates drift between the console and the state. Then they run another apply, which tries to revert the manual changes, which creates a cycle of confusion. The right response to a partial apply: read the error message, understand what failed, fix the root cause (if it is in the Terraform code), and run another plan-and-apply cycle through the normal CI process.

**Why apply-in-CI, not locally?**

Local applies bypass the audit trail. If engineer A applies locally with their personal credentials, there is no GitHub Actions log, no artifact reference, no PR linking the change to the applied plan. If something goes wrong, you cannot answer "what was applied, when, by whom, from which commit?" CI applies create an immutable audit trail: every apply links to a commit SHA, a PR number, an IAM role ARN, a timestamp, and the exact plan binary that was applied.

Local applies also use local credential context. The CI runner uses OIDC to federate to an IAM role with exactly the permissions needed for the apply. An engineer's personal IAM user may have broader permissions, making local applies riskier in terms of blast radius.

The enforce-apply-in-CI rule is implemented by making the apply IAM role assumable only by the GitHub Actions OIDC provider for your repository:

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
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:myorg/myrepo:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

The condition restricts the role assumption to the main branch of your specific repository. Not from a PR branch. Not from a fork. Only from a push to main. This means engineers can run `terraform plan` locally with a read-only role, but `terraform apply` can only succeed from the CI pipeline after a merge.

### Using -target for surgical applies

There are legitimate cases for `terraform apply -target`: when you need to apply one specific resource to unblock a deployment without running the full apply for a large state. The canonical use case is recovering from a state inconsistency where one resource is out of sync and the rest of the state is fine.

```bash
# Apply only the specific RDS security group, not the entire app layer
terraform apply -target=aws_security_group.rds_ingress -out=tfplan-targeted.binary

# Refresh only (no changes) — reconcile state file with actual cloud state
terraform apply -refresh-only -out=tfplan-refresh.binary
```

The `-refresh-only` flag deserves special attention. It runs a plan that will only update the state file to match the actual resource attributes — it will not make any changes to your real infrastructure. Use it when you suspect your state file is stale (perhaps someone made a manual change to the AWS console) and you want to bring the state in sync without applying any code changes.

**The important warning about -target:** using `-target` bypasses the full dependency graph. Terraform will not recalculate all downstream resource dependencies for the targeted resource. This can leave your state in an inconsistent intermediate state where the targeted resource is updated but its dependencies are not. Use `-target` only as a short-term tactical tool, and always follow up with a full plan-and-apply cycle to verify nothing is left inconsistent.

## 8. Comparing CI Terraform orchestration tools

Multiple tools solve the plan-in-CI problem with different trade-offs. Choosing the right one depends on your team size, existing tooling, and willingness to operate additional infrastructure.

![Matrix comparison of CI Terraform tools showing Atlantis, Terraform Cloud, GitHub Actions native, and Spacelift across plan-on-PR, approval gate, state management, and cost model dimensions](/imgs/blogs/managing-terraform-safely-at-scale-3.png)

| Tool | Plan on PR | Approval gate | State management | Cost | Best for |
|------|-----------|---------------|-----------------|------|---------|
| **Atlantis** | Built-in, PR comment | Built-in (required_approvals) | Any backend (S3, GCS) | Free, self-hosted | Teams wanting open-source, full control |
| **HCP Terraform** | Built-in | Built-in (workspace permissions) | Managed by HCP | Paid (free tier for small teams) | Teams wanting managed state + Sentinel |
| **GitHub Actions native** | Manual workflow wiring | GitHub Environments | Any backend | Pay per CI minute | Teams already in GH, low additional cost |
| **Spacelift** | Built-in | Built-in, policy-driven | Managed by vendor | Paid per run | Enterprise teams with complex approval needs |
| **Scalr** | Built-in | Built-in, RBAC-based | Managed by vendor | Paid | Teams wanting HCP alternative |
| **GitLab CI + manual jobs** | Manual wiring | GitLab Environment approvals | Any backend | Included in GitLab | Teams on GitLab |

The practical decision tree:

- **Three-person startup** → GitHub Actions native. Minimal operational overhead, good-enough safety with GitHub Environments for approval gates, no new vendor to manage.
- **10–50 engineers, multi-account AWS, open-source preference** → Atlantis. It is the gold standard for teams that want plan-on-PR, comment-driven apply (`atlantis apply`), and self-hosted state. Runs as a Kubernetes deployment.
- **50+ engineers, compliance requirements, prefer managed** → HCP Terraform. Sentinel policies, audit logging, team-based access control, and managed state are worth the cost at scale.
- **Google Cloud** → Cloud Build with a custom plan-comment step or Terraform Cloud. The same principles apply; the OIDC provider is different.

## 9. Drift detection at scale

Drift is what happens when the world disagrees with your state file. An engineer creates an S3 bucket manually "just for testing" and forgets to delete it. A colleague modifies a security group rule in the AWS console to unblock a debug session and never updates the Terraform. An AWS service automatically modifies a resource attribute that your Terraform does not manage. Over time, these small divergences accumulate into a state where your Terraform code no longer accurately describes your actual infrastructure.

The definition of drift is precise: drift is any difference between the current real-world state of a resource and what the Terraform state file and code would produce. Drift can be additive (something exists in the cloud that is not in Terraform), subtractive (something is in Terraform that was manually deleted from the cloud), or attributive (a resource exists in both, but some attribute is different).

Additive drift is the most common. It is also the most dangerous, because the next apply might try to delete the manually created resource (Terraform sees it in the cloud but not in the state, so it plans to destroy it if it was somehow created outside of Terraform's management). Attributive drift is the sneakiest — it can cause the next apply to "revert" a manual emergency fix that an engineer made at 3 AM to stop an incident.

### Drift taxonomy by cause

Understanding why drift happens helps you prevent it:

| Cause | Frequency | Prevention |
|-------|-----------|-----------|
| Emergency console changes | High | Document manual changes immediately; create a "drift PR" within 24 hours |
| Auto-modification by AWS services | Medium | Use `ignore_changes` lifecycle for known auto-modified attributes |
| Deleted resources not removed from Terraform | Medium | Code review; use `terraform state rm` to remove from state when deleting manually |
| Module version upgrades with default changes | Low | Pin module versions; review changelogs before upgrading |
| Provider version upgrades with behavior changes | Low | Pin provider versions; test upgrades in dev first |
| Terraform bugs | Very low | Pin Terraform version; test upgrades in dev first |

The `ignore_changes` lifecycle block is worth knowing. When a resource has attributes that are expected to be modified outside of Terraform (e.g., an auto-scaling group's `desired_capacity` is managed by the autoscaler, not Terraform), you can tell Terraform to ignore changes to that attribute:

```hcl
resource "aws_autoscaling_group" "web" {
  name             = "web-asg"
  min_size         = 2
  max_size         = 20
  desired_capacity = 4  # Initial value; autoscaler will modify this

  lifecycle {
    ignore_changes = [
      desired_capacity,  # Managed by the autoscaler, not Terraform
      tag,               # Tags may be added by AWS Config rules
    ]
  }
}
```

Without `ignore_changes`, every `terraform plan` will show `desired_capacity = 12 -> 4` (or whatever the autoscaler set it to), and every apply will fight the autoscaler by setting it back to 4. With `ignore_changes`, Terraform ignores that attribute during plan and apply.

### Responding to drift alerts

When drift detection reports drift, the response should be deliberate, not reactive. You have three options:

**Option 1: Bring the code in line with the actual state.** If the manual change was intentional and correct, update the Terraform code to match it and run a plan-and-apply cycle. The apply will see no drift (the code now matches reality) and the state file will be updated. This is the preferred response for emergency fixes that turned out to be the right fix.

**Option 2: Remove the drift.** If the manual change was incorrect or temporary, run a Terraform apply that reverts it. This is the preferred response for test resources that should have been deleted or security group rules added for debugging.

**Option 3: Use -refresh-only to update the state file.** If a resource attribute was changed by AWS automatically (e.g., an RDS instance's `latest_restorable_time` is updated by AWS on every backup), and you want to update the state file to reflect the current attribute value without making any code changes, use `terraform apply -refresh-only`. This tells Terraform to update its state file to match the actual cloud state, without making any changes to the actual cloud resources.

The wrong response to drift: panic, run `terraform apply` without reviewing the plan, and inadvertently revert a critical emergency fix that was keeping production running.

The danger of drift is not the individual drifted resource. It is the compounding effect. Each drifted resource adds uncertainty to your next plan output. Terraform may generate a plan that wants to revert the manual change — which might be right or might break something. The larger the drift, the less you trust your plan output, the less frequently engineers run `terraform apply`, and the more drift accumulates. It is a feedback loop with only one direction: worse.

![Timeline showing the safe Terraform apply lifecycle from PR open through plan, OPA check, plan comment, human approval, merge plus apply, and nightly drift detection](/imgs/blogs/managing-terraform-safely-at-scale-5.png)

**Measuring drift.** The `terraform plan` exit code tells you: 0 = no changes needed (no drift), 2 = changes needed (drift detected), 1 = error. A scheduled drift detection job runs `terraform plan` against each state file and alerts if it exits with code 2.

```yaml
# .github/workflows/terraform-drift.yml
name: Terraform Drift Detection

on:
  schedule:
    - cron: "0 6 * * 1-5" # 6 AM UTC, weekdays
  workflow_dispatch: # Allow manual trigger

jobs:
  drift-check:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        layer: [foundation, networking, platform, app]
      fail-fast: false # Check all layers even if one has drift

    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.8.5"

      - name: Configure AWS credentials (read-only OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-ci-plan-role
          aws-region: us-east-1

      - name: Terraform Init
        working-directory: infra/${{ matrix.layer }}
        run: terraform init

      - name: Check for drift
        id: drift
        working-directory: infra/${{ matrix.layer }}
        run: |
          set +e
          terraform plan -detailed-exitcode -out=/dev/null 2>&1
          EXIT_CODE=$?
          set -e
          if [ "$EXIT_CODE" -eq "2" ]; then
            echo "drift=true" >> "$GITHUB_OUTPUT"
            echo "DRIFT DETECTED in layer: ${{ matrix.layer }}"
          elif [ "$EXIT_CODE" -eq "1" ]; then
            echo "drift=error" >> "$GITHUB_OUTPUT"
            exit 1
          else
            echo "drift=false" >> "$GITHUB_OUTPUT"
            echo "No drift in layer: ${{ matrix.layer }}"
          fi

      - name: Notify on drift
        if: steps.drift.outputs.drift == 'true'
        uses: actions/github-script@v7
        with:
          script: |
            // Post a GitHub issue or send a Slack notification
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Terraform drift detected: ${{ matrix.layer }} layer`,
              body: `Drift was detected in the \`${{ matrix.layer }}\` Terraform layer during the scheduled drift check.\n\nRun \`terraform plan\` in \`infra/${{ matrix.layer }}\` to see the differences.\n\nCommit: ${context.sha}`,
              labels: ['terraform-drift', 'infrastructure']
            });
```

**Drift that caused the incident: a realistic scenario.** A team's networking layer has a security group that controls ingress to the RDS cluster. Three months ago, an engineer added a manual ingress rule in the AWS console to debug a connectivity issue from a VPN client. She intended to remove it after the debug session, forgot, and moved on. The security group now has one rule in the console that does not exist in Terraform.

The team runs a routine Terraform apply to add a new security group for a new service. Terraform plans the apply: it will add the new security group and also remove the manual ingress rule from the existing one (because the rule is not in the Terraform code). The engineer reviewing the plan sees `# aws_security_group_rule.rds_ingress_debug: will be destroyed` and — because the plan has 40 lines and they are in a hurry — approves it.

The apply runs. The manual ingress rule is removed. Within ten minutes, the VPN-connected data analyst running end-of-quarter reports gets an access denied error. Four engineers spend two hours debugging a connectivity issue that could have been caught by a drift detection alert three months ago.

Daily drift detection would have flagged `drift=true` in the networking layer the day after the manual rule was added. A GitHub issue would have been created. An engineer would have either removed the rule or added it to Terraform code. The incident would never have occurred.

## 10. The blast-radius comparison: before and after layering

![Blast-radius comparison showing a monolith state where one apply can touch 200 resources versus layered state files where an apply targets 30 resources in one layer](/imgs/blogs/managing-terraform-safely-at-scale-6.png)

The blast-radius reduction is visible in monitoring data. Here is what a team moving from monolith to layered state typically observes, expressed as before→after metrics (these are illustrative order-of-magnitude numbers based on commonly reported patterns, not a specific company benchmark):

| Metric | Before (monolith) | After (layered) | Change |
|--------|------------------|-----------------|--------|
| Plan time (average) | 9.5 min | 85 sec | -91% |
| Resources per apply (max) | 247 | 38 | -85% |
| Change-failure rate (Terraform applies) | 17% | 3% | -82% |
| Mean time to apply a change | 45 min (including wait for 9-min plan) | 6 min | -87% |
| Engineers making infra PRs/month | 4 | 12 | +200% |
| Terraform-caused incidents/quarter | 2.3 (average) | 0.3 (average) | -87% |

The last row is the one that matters for DORA. The change-failure rate for infrastructure went from 17% to 3%. That is a massive improvement in delivery performance, and it came entirely from structural changes to how state was organized — not from better code, not from more experienced engineers, not from a new tool.

## 11. The Terraform safety controls taxonomy

Every Terraform safety practice falls into one of three phases: pre-apply (preventing bad plans from being approved), during-apply (limiting damage from an apply already in flight), and post-apply (detecting and remedying problems that slipped through).

![Tree diagram of Terraform safety controls taxonomy with root safety controls branching into pre-apply, during-apply, and post-apply phases, each with specific leaf controls](/imgs/blogs/managing-terraform-safely-at-scale-7.png)

**Pre-apply controls** are the cheapest. A plan review that catches a forced replacement costs 5 minutes. Preventing the apply that would have caused a 4-hour outage costs the 5 minutes. OPA policy that blocks a public S3 bucket takes milliseconds to run. Preventing the bucket that would have triggered a compliance incident is free.

**During-apply controls** are for when something slips through. `-target` limits an apply in flight to specific resources. The approval gate in GitHub Environments is the last human checkpoint. The CI bot's scoped IAM role is the last permission checkpoint.

**Post-apply controls** are your safety net. Drift detection catches manual changes before they accumulate. State backups (S3 versioning on the state bucket) let you roll back a corrupted state. Cost alerts (via AWS Cost Explorer + billing alarms defined in the `foundation` layer) catch runaway resource creation.

The most common mistake is investing only in during-apply controls (approval gates) without pre-apply checks (OPA policies) or post-apply monitoring (drift detection). An approval gate is only as good as the human reviewer, and humans miss things. Defense in depth means all three phases have controls.

## 12. War story: the Terraform apply that wanted to destroy prod

This is a composite of real patterns I have seen. The names are fictional; the failure mode is not.

**The setup.** A mid-size e-commerce company has 180 resources in a single Terraform state. Their CI pipeline runs `terraform apply -auto-approve` on every merge to main. No plan review. No policy checks. No approval gate. They have been operating this way for two years with no major incidents, which has given them a false sense of security.

**The trigger.** An engineer is adding a new CloudFront distribution for a new landing page. The module they use depends on an ACM certificate. They add the certificate resource to the Terraform code. During local testing, they notice the certificate has `create_before_destroy = true` in the lifecycle block (inherited from the module defaults) — this is fine. They do not notice that the `ssl_support_method` argument on the existing CloudFront distribution (which serves the main storefront) is also being set by the same module with a different default value than what was manually set in the console two years ago.

**The plan.** The CI pipeline runs `terraform apply -auto-approve`. The plan (which no human reviews, because auto-approve) shows `# aws_cloudfront_distribution.storefront: must be replaced`. The engineer is working on a different PR. Nobody sees the plan output.

**The apply.** Terraform begins destroying the production CloudFront distribution. CloudFront distributions take 15–20 minutes to delete and 15–20 minutes to recreate. The main storefront is down for 34 minutes. A Friday afternoon. The team is paged. The sales team, mid-quarter, is watching revenue drop in real time.

**The cause.** Not Terraform's fault. Not the engineer's fault (they did not know their change would affect the existing distribution). The fault is architectural: no plan review gate, no policy to alert on large-scale destroy operations, no blast-radius limit that would have isolated the CloudFront distribution in a separate state from the new certificate.

**The remediation.** The team implemented all four safeguards covered in this post within the following sprint: plan-in-CI with PR comment, manual approval for any plan showing destroys, OPA policy blocking any plan with more than 5 resource destructions without explicit override, and state layering that separated the CloudFront distributions into an `edge` state file separate from the `app` state.

**The outcome.** Change-failure rate for Terraform applies dropped from 22% to 2% over the following two quarters. Plan time for the `app` layer dropped from 11 minutes to 2 minutes. The team went from running Terraform twice a week (to reduce the number of potentially dangerous applies) to running it multiple times per day.

## 13. Stress-testing the safe Terraform workflow

Before trusting any pipeline design, it is worth asking: what happens when it breaks? Here are the scenarios that trip up well-designed Terraform CI pipelines, and how to handle them.

**What if the plan artifact is corrupted or lost?** GitHub Actions artifacts have a maximum retention period (default: 90 days). If the apply job tries to download an artifact that no longer exists, it fails. Implement a fallback: if the artifact download fails, re-run the plan step (with the same environment and commit SHA), post the new plan as a comment, and require re-approval before proceeding. Alternatively, store the plan binary in the S3 state bucket (alongside the state file) with the commit SHA as the object key.

**What if two PRs merge at the same time?** The first apply job acquires the state lock and runs. The second apply job waits for the lock (with `-lock-timeout=10m`). When the first apply completes and releases the lock, the second apply runs. The second apply uses its own saved plan binary. If the first apply changed resources that the second plan expected to be in a certain state, Terraform will error when it verifies the pre-apply state. This is the correct behavior — it prevents the second apply from proceeding with a stale plan.

**What if the CI bot's OIDC token expires mid-apply?** AWS OIDC tokens are valid for a maximum of 12 hours; the default is 1 hour. A large apply that takes 90 minutes will have its token expire mid-run. Set the token duration to 2 hours for apply roles:

```yaml
- name: Configure AWS credentials (apply role, 2-hour token)
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::123456789012:role/terraform-ci-apply-role
    role-duration-seconds: 7200
    aws-region: us-east-1
```

**What if an OPA policy blocks a legitimate change?** Every policy needs an override mechanism for exceptional cases. For OPA/Conftest, implement an override via a file in the PR: if a file named `policy-override.json` exists with the correct format and a required field explaining the reason, the policy check passes. The override file is committed to the repository, creating an audit trail. Require a separate approval from the security team for any PR that includes a policy override file.

**What if Terraform state becomes corrupted?** This is rare but real. If the state file is corrupted (e.g., from a failed apply that wrote partial state), you need the S3 versioning on the state bucket. Recover by restoring the previous version of the state file: `aws s3api restore-object ...` or navigate to the S3 bucket in the console and restore a previous version. Always enable S3 versioning on state buckets. This is not optional.

**What if drift detection runs during an active apply?** If the nightly drift detection cron runs at 6 AM and an engineer's apply is running at 6 AM, the drift check will fail to acquire the state lock and produce an error. Set drift detection to not fail the workflow on lock contention — treat lock contention as "apply in progress, try again later," not as an error:

```bash
set +e
terraform plan -detailed-exitcode -lock-timeout=2m -out=/dev/null 2>&1
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -eq "1" ]; then
  # Check if the error is a lock contention
  if terraform state list 2>&1 | grep -q "state is locked"; then
    echo "State locked by another process — skipping drift check"
    exit 0
  fi
  exit 1
fi
```

## 14. How to reach for this, and when not to

These practices are powerful, but they are also costs. Not every team needs every layer.

**When to adopt plan-in-CI with PR comment:** almost always, even for a solo engineer. The overhead is one GitHub Actions workflow and an OIDC role. The benefit — seeing your plan before it runs — is immediate. Do this first.

**When to add OPA/Conftest policies:** when your team has more than three engineers making infrastructure changes, or when you have compliance requirements (SOC 2, PCI-DSS, HIPAA) that mandate specific controls. Start with 3–5 policies covering the highest-risk patterns (public storage, overly permissive security groups, missing required tags) and expand from there.

**When to layer your state:** when your plan time exceeds 3 minutes (a symptom of > 80 resources in a state), when you have teams with different deploy cadences (app engineers deploy daily, network engineers deploy quarterly), or when a single erroneous apply could affect your entire production infrastructure.

**When NOT to layer your state:** when you have fewer than 50 resources and a single-person team. Layering adds operational complexity (remote state data sources, separate backend configs, separate CI workflows for each layer). A 3-person startup with 40 resources is better served by a single state with plan-in-CI and an approval gate.

**When to add drift detection:** when your infrastructure is shared across teams, when you have on-call engineers who make emergency manual changes, or when you have found drift-caused incidents in your postmortems. Daily or weekly drift detection is a good habit for any team operating infrastructure at scale.

**When not to use -target:** never use `-target` as a routine workflow. It is a surgical tool for emergency situations. If you find yourself using `-target` regularly to avoid applying "the parts of the state you don't trust," that is a signal that you need to layer your state, not that you should keep using `-target` as a workaround.

**The Atlantis vs HCP Terraform decision.** Do not pay for HCP Terraform if you are running fewer than five workspaces and do not need Sentinel or managed state. Atlantis is free, open-source, and covers 90% of what teams need. Pay for HCP Terraform when you need Sentinel hard-mandatory policies (compliance), when you want managed state and run summaries without building your own, or when your team has 50+ workspaces and the operational overhead of Atlantis becomes a bottleneck.

## 14. Pulling it together: a complete safe Terraform workflow

Here is the full picture of what a safe Terraform workflow looks like, with all layers active.

**Day-to-day flow for a Terraform change:**

1. Engineer writes HCL, opens a PR against the `infra/app` directory.
2. `terraform-plan.yml` triggers automatically: `terraform init` → `fmt -check` → `validate` → `plan -out=tfplan.binary` → convert to JSON → upload artifact → post comment on PR. Elapsed time: ~90 seconds.
3. `conftest test tfplan.json --policy policies/` runs as part of the plan workflow. If any Rego rule fires, the CI fails and the PR is blocked.
4. An engineer (or the PR author if the plan shows zero destroys and the policy says the plan is safe) reviews the PR comment showing the plan output. They check the summary line for destroys, look for `-/+` symbols, and verify the resource count is what they expected.
5. The engineer approves the PR and merges to main.
6. `terraform-apply.yml` triggers. The `environment: production` gate requires a second approval from a GitHub Environment reviewer for any apply to production.
7. The apply step downloads the exact `tfplan.binary` artifact from step 2 (same commit SHA). It runs `terraform apply tfplan.binary` — no new plan, no fresh API calls that might diverge.
8. The state is updated. The apply job outputs a link to the run log, which is attached to the merge commit.

**Nightly:**

The drift detection workflow runs `terraform plan -detailed-exitcode` against all four layers. Any exit code 2 creates a GitHub Issue and sends a Slack notification to the infrastructure team channel.

**Monthly:**

Review the OPA policy violations log. Are engineers regularly triggering the instance-type policy? Perhaps the approved list needs updating. Are required-tag violations frequent? Perhaps the module defaults need updating to include the required tags automatically.

![Plan review before-after showing plan output skipped unread leading to forced replace and DB destruction versus careful review catching the forced replace before apply preventing outage](/imgs/blogs/managing-terraform-safely-at-scale-8.png)

#### Worked example: catching a forced replacement before it destroys the database

An engineer is changing the `db_parameter_group_name` on an RDS cluster from the default parameter group to a custom one. The Terraform plan shows:

```bash
  # aws_db_instance.production must be replaced
-/+ resource "aws_db_instance" "production" {
      ~ db_parameter_group_name = "default.mysql8.0" -> "custom-mysql-prod" # forces replacement
        id                      = "production-mysql-cluster"
        # (47 unchanged attributes hidden)
    }

Plan: 0 to add, 0 to change, 1 to destroy. 1 to add, 0 to change, 0 to destroy.
```

The `# forces replacement` annotation is Terraform's signal. The engineer reading this plan sees `-/+` on the RDS instance and the words `forces replacement`. They know this means: delete the current RDS instance, create a new one. That is a database deletion event.

Without plan review: the engineer sees "Plan: 0 to add, 0 to change, 1 to destroy" in the summary and misreads it (thinking it is something unimportant) and approves. The apply destroys the production database. MTTR: 4 hours (restore from backup, update connection strings, restart services).

With plan review: the engineer sees the `-/+` symbol and the `forces replacement` annotation. They stop and investigate. They learn that `db_parameter_group_name` changes require a blue-green deployment, not an in-place update. They update the Terraform code to use a `blue_green_update` lifecycle block or schedule a maintenance window with a proper database migration procedure.

The five-minute plan review prevented a four-hour incident.

## Key takeaways

1. **Plan-in-CI is the first and most impactful safety layer.** Every PR that touches Terraform should trigger an automatic plan, with the output posted as a PR comment. This eliminates the "what will this actually do?" uncertainty from the apply step.

2. **Plan once, apply exactly that plan.** Save the plan binary with `-out`, upload it as a CI artifact, and pass the artifact to the apply step. Never let a fresh plan run implicitly in the apply step — it may produce a different result.

3. **Blast radius is a function of state size.** Split monolith states into layers (foundation, networking, platform, app). Each layer should have 20–50 resources. Plan time drops from minutes to seconds; per-apply blast radius drops by 10×.

4. **Policy-as-code is not optional at team scale.** Write 5–10 OPA/Conftest policies covering your highest-risk patterns and run them against the plan JSON on every PR. They catch what humans miss.

5. **The `-/+` symbol is the most dangerous thing in a Terraform plan.** Train every engineer who reviews plans to stop on any `-/+` symbol and understand why the replacement is forced before approving.

6. **Apply in CI, never locally.** Use OIDC federation to give CI a scoped IAM role that is not assumable by local credentials. Make CI the only place that can run apply against production state.

7. **Drift compounds.** A single undetected manual change is not a disaster. Ten undetected manual changes across six months is an incident waiting to happen. Run drift detection daily; treat drift alerts as P2 issues to remediate within a week.

8. **Use `-target` only in emergencies.** It bypasses the dependency graph and can leave your state inconsistent. If you use it, always follow up with a full plan-and-apply cycle.

9. **Layered state enables independent deploy cadence.** The app layer can change ten times a day; the foundation layer changes once a quarter. Layering lets each team move at their own speed without stepping on each other.

10. **Defense in depth beats single-point controls.** Approval gates fail when reviewers are tired. OPA policies fail when a case was not anticipated. Drift detection catches what both miss. Layer all three.

## Further reading

- [The CI/CD mental model: commit to production](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the full pipeline spine this post builds on
- [Infrastructure as code: from ClickOps to declarative](/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative) — the foundational IaC concepts and why declarative infrastructure beats imperative shell scripts
- [Terraform in practice: state, modules, and workspaces](/blog/software-development/ci-cd/terraform-in-practice-state-modules-and-workspaces) — the mechanics of Terraform state and module design that this post's safety practices build on
- [The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — how to use error budgets to decide how aggressively to automate infrastructure changes and when to slow down
- [Terraform documentation: Plan command](https://developer.hashicorp.com/terraform/cli/commands/plan) — the official reference for all plan flags including `-out`, `-target`, `-refresh-only`, and `-detailed-exitcode`
- [OPA/Conftest documentation](https://www.conftest.dev/) — policy testing for configuration files, including Terraform plan JSON
- [HashiCorp Sentinel documentation](https://developer.hashicorp.com/sentinel) — the policy-as-code framework for HCP Terraform
- [Atlantis documentation](https://www.runatlantis.io/) — the open-source Terraform pull request automation tool
- [DORA State of DevOps Report](https://dora.dev/) — the foundational research on delivery performance metrics that grounds every safety practice in measurable outcomes
