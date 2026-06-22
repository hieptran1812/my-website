---
title: "Infrastructure as code: from ClickOps to declarative"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn why ClickOps creates snowflake servers nobody can reproduce, how declarative IaC solves it with a plan/apply reconcile loop, and how to treat infrastructure as a versioned delivery artifact from day one."
tags:
  [
    "ci-cd",
    "devops",
    "infrastructure-as-code",
    "terraform",
    "iac",
    "declarative",
    "drift",
    "platform-engineering",
    "gitops",
    "cloud",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-1.png"
---

It was a Tuesday afternoon when the on-call engineer's phone lit up. The production web servers were responding with `502 Bad Gateway`. The load balancer health checks were failing. Eighteen months of gradual, manual tuning — a security group rule tweaked here, a kernel parameter changed there, a package installed at 2 AM during an outage — had made the three "identical" servers quietly different from each other. When a scheduled patch reboot hit Server B, it came back up in a subtly different state than Server A or Server C. Nothing was documented. Nobody knew what the expected state even was anymore. The recovery took eleven days because the team had to reverse-engineer a working configuration by diffing `/etc/` directories across the three machines, hunting through bash history, and piecing together a year and a half of undocumented changes. When they finally rebuilt the server, they still weren't sure they'd gotten it right.

That story — the snowflake server, the undocumented change, the eleven-day rebuild — is the reason Infrastructure as Code exists. The figure below shows the before and after in stark terms: on the left, manually-clicked servers diverge over time into unique, unreproducible configurations; on the right, every server in the fleet is identical because they were all created from the same version-controlled source of truth.

![ClickOps produces unique unreproducible snowflake servers, while IaC delivers an identical auditable fleet on every apply](/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-1.png)

By the end of this post you will understand why ClickOps is structurally incapable of producing reproducible infrastructure, how the declarative plan/apply reconcile model fixes this, how to treat infrastructure as a delivery artifact that lives in Git alongside your application code, what drift is and why it is the silent enemy of every IaC deployment, and when IaC is genuinely not worth the overhead. You will walk away with working Terraform HCL for a simple VPC and EC2 instance, a GitHub Actions workflow that runs `terraform plan` as a CI gate, a concrete war story about a Terraform apply that wanted to destroy the production database, and a clear decision framework for when to reach for IaC vs. when to stay on a PaaS.

This post is part of the "CI/CD & Cloud-Native Delivery" series. The commit→build→test→package→deploy→operate spine applies here too: infrastructure is a delivery artifact just like a Docker image, and IaC is how you move it through a reproducible pipeline. The DORA metric most affected by IaC is **lead time for changes** — the time from "we need a new environment" to "that environment is live" drops from weeks to hours. We'll measure that concretely.

---

## The ClickOps problem: anatomy of a snowflake server

ClickOps is not a derogatory term for laziness. It's a description of a workflow that feels entirely reasonable when you're starting out: you log into the AWS console, click through the wizard, set up your VPC, launch your EC2 instance, SSH in, run `apt install nginx`, tweak the config, open the security group ports you need, and you're done. It works. The server serves traffic. You move on.

The problem does not show up on day one. It shows up over eighteen months of accumulated decisions made under time pressure.

Here is how a snowflake is born. It starts with a Tuesday at 11 PM: the server is running hot because the app deployed a memory-hungry new feature. You SSH in, bump the JVM heap size in `/etc/systemd/system/app.service`, and restart. It works. You don't write it down because it's 11 PM and the incident is over and you'll "document it later." A month later the intern spins up a second server by copying the original — except the original was the version from three months ago because the AMI was built then, so the heap setting is not there. Another month passes. A security audit finds port 8080 is open on Server A from an old debugging session. Someone closes it. Server B still has it open because that server was set up by a different person from a different AMI with different defaults. The divergence compounds. No single change was irresponsible. The problem is architectural: the console is the worst possible place to store configuration because it has no version history, no diff, no rollback, no test gate, and no single source of truth.

### The four symptoms of a ClickOps environment

**Symptom 1 — Unreproducibility.** If Server A dies, you cannot recreate it exactly. You can get close, but "close" means "different state" means "potential behavior difference under load." The only way to know what's on a server is to log in and look.

**Symptom 2 — Configuration drift between instances.** Every server in a ClickOps fleet diverges from every other over time. The divergence is not random noise — it is the accumulated record of every incident, every shortcut, every "I'll fix it properly later." By twelve months in, no two servers are the same, and nobody can explain why.

**Symptom 3 — Tribal knowledge dependency.** The system's configuration lives in the heads of the people who made the changes. When those people leave, the knowledge leaves with them. This is not a hiring or retention problem. It is a tooling problem. The tool (the console) does not record decisions; therefore decisions are not recorded.

**Symptom 4 — Inability to create a new environment quickly.** Need a staging environment that mirrors production? That is now a multi-week project requiring someone to document production, recreate it from memory, and verify the result — a verification that is itself unreliable because there is no authoritative definition of what production actually is.

The Friday-afternoon console hero deserves special attention. This is the engineer who notices production is slow at 4:45 PM on a Friday, opens the console, changes the instance type, bumps a parameter, closes a security group rule that looked suspicious, and goes home for the weekend. On Monday, production is working fine. Nobody knows what changed. Nobody can revert it cleanly. Nobody knows whether that security group change was correct or whether it accidentally opened a different attack surface. Two months later, during a compliance audit, someone asks "why is port 3306 open to 0.0.0.0/0 on that instance?" Nobody knows. The console hero is not malicious. The console is the footgun.

### The economics of snowflake servers

The cost of ClickOps is not just the rebuilding time. It compounds through the entire delivery lifecycle. Here is a rough accounting of where ClickOps actually hurts your team:

**Onboarding cost.** Every new engineer who joins the team has to learn the production environment by walking around it — SSH-ing in, reading config files, asking veterans "why is this set to that?" This is not knowledge transfer. It is archaeological excavation. In a ClickOps environment, onboarding an engineer to the point where they can safely make infrastructure changes reliably takes four to six weeks. In an IaC environment, a new engineer can open the repository, run `terraform plan`, and understand the complete picture of production infrastructure in an afternoon.

**Debugging cost.** When a production incident involves infrastructure, the first question is always "what changed?" In a ClickOps environment, answering that question requires reviewing AWS CloudTrail (which shows API calls but not intent), interviewing team members, and reviewing bash history — all under incident pressure. In an IaC environment, the answer is `git log infra/` plus the GitHub PR history. Every change is annotated with who made it, why, and what was reviewed.

**Compliance cost.** Financial services, healthcare, and any company handling PII faces auditors who want to see change management processes for infrastructure. "We click things in the console when we need to" is not a change management process. IaC with PR review and CI gates is. Companies in regulated industries that adopt IaC typically find that their next SOC 2 or ISO 27001 audit goes substantially faster because the evidence is already there: Git history, PR approvals, CI pipeline logs.

**Scaling cost.** ClickOps does not scale. At five servers managed by two engineers, it is manageable. At fifty servers managed by ten engineers — each with their own preferences, their own style, their own "I'll document it later" — it becomes chaos. IaC scales linearly: adding a new server is a PR that references an existing module. The incremental cost of managing the hundredth server is nearly zero once the module exists.

**Recovery cost.** When disaster strikes — a region failure, an accidental deletion, a ransomware attack — the question is how long it takes to restore production. With ClickOps, the answer is measured in days or weeks because you are rebuilding from memory and tribal knowledge. With IaC, the answer is measured in hours because you are running a pipeline against a complete, versioned specification of production infrastructure.

The aggregate economic case is strong. Teams that adopt IaC early consistently report 3x–5x reductions in infrastructure-related incident resolution time, 10x–30x reductions in new environment lead time, and significant reductions in compliance audit preparation overhead. These are not marketing figures — they are the kind of numbers that show up in the DORA State of DevOps research as correlates of elite delivery performance.

---

## Declarative vs imperative IaC: the core distinction

The word "Infrastructure as Code" covers a wide range of tools that work quite differently. The most important distinction is between **declarative** and **imperative** approaches, and understanding this distinction is the key to understanding why tools like Terraform solve the snowflake problem while tools like Ansible or shell scripts only partially address it.

![IaC approaches split into declarative desired-state tools and imperative command-sequence tools with fundamentally different guarantees](/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-7.png)

### Imperative IaC: "do this"

An imperative IaC tool executes a sequence of commands. A shell script that provisions a server is imperative: it runs `apt update`, then `apt install nginx`, then `cp nginx.conf /etc/nginx/nginx.conf`, then `systemctl restart nginx`. A simple Ansible playbook is largely imperative in practice, even though Ansible tasks have some idempotency built in (the `creates:` parameter, the `when:` conditional).

The core problem with imperative IaC is that it describes a **procedure**, not a **desired state**. If you run the script twice, on the second run it will try to install nginx again, restart a service that may already be running, and potentially overwrite configuration that was intentionally modified after the first run. This is called **non-idempotency**: the same input does not produce the same output when applied multiple times. You can work around this with conditionals (`if nginx is not installed, install it`), but now you are manually implementing a desired-state check — which is what declarative tools give you for free.

More critically, imperative tools have no concept of the **current state** of infrastructure. They know what commands to run; they do not know what already exists. If you provision an EC2 instance with a script, and then run the script again on a fresh machine, it will provision another instance. There is no "the infrastructure should look like this" — there is only "run these commands."

### Declarative IaC: "it should look like this"

A declarative tool like Terraform works differently. You write a description of the infrastructure you want:

```hcl
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"

  tags = {
    Name        = "web-server"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}
```

This is not a script. This is a **declaration of desired state**: "I want an EC2 instance with these properties." Terraform's job is to read this declaration, compare it to the current state of AWS, compute the diff, and apply only the changes needed to bring reality in line with the declaration.

This is the **reconcile loop model**: 

```
desired state (your .tf files)
        ↓
   terraform plan (compute diff against current state)
        ↓
   terraform apply (apply only the delta)
        ↓
current state ≈ desired state
```

The key property this gives you is **idempotency**: you can run `terraform apply` ten times, and if nothing has changed in your `.tf` files, nothing will change in your infrastructure. The tool knows what already exists. It tracks this in a **state file** (`terraform.tfstate`) that records the mapping between your declared resources and the real cloud resources they correspond to.

### Why idempotency matters for delivery

Idempotency is not just a technical nicety. It is what makes infrastructure safe to put in a CI/CD pipeline. When a CI pipeline runs `terraform apply` on every merge to main, you need to be confident that:

1. If nothing changed in the `.tf` files, nothing will change in production (no accidental side effects).
2. If something did change, only exactly that thing will change (minimal blast radius).
3. The same pipeline can be run by any engineer on any machine and will produce the same result (no "works on my laptop" IaC).

A shell script fails all three. A declarative IaC tool satisfies all three by design.

| Property | Shell scripts | Ansible | Terraform / Pulumi |
|----------|--------------|---------|-------------------|
| Idempotent by default | No | Partial | Yes |
| Tracks current state | No | No | Yes (state file) |
| Computes minimal diff | No | No | Yes (plan) |
| Safe to run in CI | No | Risky | Yes |
| Detects drift | No | No | Yes (refresh) |
| Language | Bash / Python | YAML | HCL / real languages |

---

## The plan/apply model: reading the diff before you destroy the database

The single most important safety mechanism in Terraform — the one that distinguishes it from a deploy script and from the AWS console — is the **plan**. Understanding the plan is not optional. It is the reason IaC is safe enough to put in production pipelines.

![Terraform plan-apply reconcile loop branches on plan safety to either human approval or abort, never proceeding blindly](/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-3.png)

### What a plan actually is

`terraform plan` performs three steps:

1. **Refresh**: reads the current state of every resource in your state file from the actual provider APIs (AWS, GCP, Azure, etc.) to detect any drift.
2. **Diff**: compares the refreshed current state against the desired state described in your `.tf` files.
3. **Output**: prints a structured diff showing exactly which resources will be created (+), destroyed (-), updated in-place (~), or replaced (destroy then create).

The plan output looks like this:

```bash
# Terraform plan output example
Terraform will perform the following actions:

  # aws_instance.web will be updated in-place
  ~ resource "aws_instance" "web" {
        id            = "i-1234567890abcdef0"
      ~ instance_type = "t3.medium" -> "t3.large"
        tags          = {
            "Environment" = "production"
            "Name"        = "web-server"
        }
    }

  # aws_security_group.db will be destroyed
  - resource "aws_security_group" "db" {
      - id                     = "sg-0abc123def456" -> null
      - name                   = "db-access"        -> null
      ...
    }

Plan: 0 to add, 1 to change, 1 to destroy.
```

See that last line? **1 to destroy.** That is the line that saves production databases.

### The "wanted to destroy 47 resources" story

Here is a scenario that has happened to teams more often than anyone likes to admit. An engineer is refactoring the Terraform module for the database tier. They move the `aws_db_instance` resource from `module.db` to `module.rds_cluster`. The name change is purely organizational — the actual RDS instance should not change at all. But Terraform sees it differently: the resource at `module.db.aws_db_instance.main` has been deleted from the configuration (that identifier no longer exists), and a new resource at `module.rds_cluster.aws_db_instance.main` has been added.

The plan output:

```bash
  # module.db.aws_db_instance.main will be destroyed
  - resource "aws_db_instance" "main" {
      - id                     = "myapp-prod-db" -> null
      - identifier             = "myapp-prod-db" -> null
      - allocated_storage      = 500             -> null
      ...
    }

  # module.rds_cluster.aws_db_instance.main will be created
  + resource "aws_db_instance" "main" {
      + identifier = "myapp-prod-db"
      ...
    }

Plan: 1 to add, 0 to change, 1 to destroy.
```

If you apply this plan, Terraform will delete your production database and create a new, empty one. The right fix is a `terraform state mv` to rename the resource in the state file without touching the actual infrastructure. But you only know to do that fix if you **read the plan**.

In practice, the plan is what separates "IaC practitioner" from "IaC user." Beginners skip the plan and run `terraform apply -auto-approve`. Practitioners treat the plan output as a mandatory artifact that must be reviewed — and they gate their CI pipelines so the plan output is posted to every PR for human review before any apply is allowed.

### Separating plan from apply in CI

The correct CI workflow for Terraform looks like this:

```yaml
# .github/workflows/terraform.yml
name: Terraform CI

on:
  pull_request:
    paths:
      - "infra/**"
  push:
    branches:
      - main
    paths:
      - "infra/**"

jobs:
  plan:
    name: Terraform Plan
    runs-on: ubuntu-latest
    permissions:
      id-token: write   # For OIDC to AWS
      contents: read
      pull-requests: write

    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-ci
          aws-region: us-east-1

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.8.0"

      - name: Terraform Init
        run: terraform init -backend-config=backend.hcl
        working-directory: infra/

      - name: Terraform Format Check
        run: terraform fmt -check -recursive
        working-directory: infra/

      - name: Terraform Validate
        run: terraform validate
        working-directory: infra/

      - name: Terraform Plan
        id: plan
        run: terraform plan -out=tfplan -no-color 2>&1 | tee plan.txt
        working-directory: infra/

      - name: Post Plan to PR
        uses: actions/github-script@v7
        if: github.event_name == 'pull_request'
        with:
          script: |
            const fs = require('fs');
            const plan = fs.readFileSync('infra/plan.txt', 'utf8');
            const truncated = plan.length > 60000 ? plan.slice(-60000) + '\n...(truncated)' : plan;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '## Terraform Plan\n```\n' + truncated + '\n```'
            });

  apply:
    name: Terraform Apply
    runs-on: ubuntu-latest
    needs: plan
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment: production   # Requires manual approval in GitHub Environments

    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-apply
          aws-region: us-east-1
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.8.0"
      - name: Terraform Init
        run: terraform init -backend-config=backend.hcl
        working-directory: infra/
      - name: Terraform Apply
        run: terraform apply -auto-approve
        working-directory: infra/
```

Notice the key structural decisions:

- **Plan runs on every PR**: the diff is visible before any code is merged.
- **Apply only runs on `main` push**: no branch can accidentally apply.
- **`environment: production`**: GitHub's deployment environment feature adds a required-reviewer gate before apply runs.
- **OIDC to AWS**: the CI runner never holds long-lived AWS credentials (covered in depth in the secrets management posts in this series).
- **Plan and apply are separate jobs**: the plan output is an artifact; the apply reads state fresh from the backend.

![The IaC delivery pipeline layers Git through CI plan gate through human review through apply to live infra with drift monitoring](/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-2.png)

### What a plan review checklist looks like

When a Terraform PR lands in your queue with a plan comment, you are not just reviewing the code — you are reviewing the plan. Here is the checklist a senior infrastructure engineer works through before approving:

**1. Count the actions.** The summary line `Plan: X to add, Y to change, Z to destroy` is the first thing you read. Any destroys require explanation. More than five destroys on a routine change is a red flag — understand each one before approving.

**2. Inspect every destroy.** Read the full diff for each resource marked for destruction. Ask: is this intentional? Is this data-bearing (databases, volumes, S3 buckets)? If it is a database or a persistent storage resource, the plan should almost never include a destroy — there should be a `prevent_destroy = true` lifecycle rule instead.

**3. Check for replacement (destroy + create).** Some changes cannot be applied in-place and force a resource replacement. The plan marks these with `-/+` rather than just `-`. A replaced EC2 instance is probably fine. A replaced RDS instance is almost certainly not (it means the old instance is destroyed and a new empty one is created — your data is gone unless you have a backup restore step planned).

**4. Verify names and tags.** Are all resource names following the team's naming convention? Are the standard tags present? Does the environment variable resolve to the correct value? Naming mistakes in IaC are permanent — Terraform tracks resources by their logical name in the configuration, so renaming a resource after creation requires a `terraform state mv` operation.

**5. Check for scope creep.** Is the plan doing more than the PR description says it should? If the PR says "add a new security group rule" and the plan shows five resource changes, read all five. Common causes of scope creep: the engineer also changed a `provider` version (which can trigger plan changes for many resources), edited a `locals` block that feeds many modules, or accidentally modified a `main.tf` in a shared module.

**6. Look for security regressions.** Security group ingress rules open to `0.0.0.0/0`, IAM policies with `*` actions or resources, public S3 bucket policies, RDS instances without deletion protection — these show up in the plan diff and can be caught here before they reach production. Checkov and tfsec catch most of these automatically; the human review catches the edge cases that automated tools miss.

This six-step review process takes two to five minutes for a typical infrastructure PR. It is the highest-leverage safety checkpoint in the IaC delivery pipeline.

---

## IaC as a delivery artifact: infrastructure lives in Git

The phrase "everything as code" — one of the two governing principles of this series alongside "build once, promote everywhere" — applies with full force to infrastructure. This is not a metaphor. Your infrastructure configuration should live in a Git repository, go through pull request review, trigger a CI pipeline, have automated tests, be versioned with tags, and be deployed through the same kind of controlled release process as application code.

### What "infra in Git" actually means

A mature IaC repository structure looks like this:

```bash
infra/
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── ec2-cluster/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── rds/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── environments/
│   ├── staging/
│   │   ├── main.tf          # composes modules
│   │   ├── variables.tf
│   │   └── backend.hcl      # remote state config
│   └── production/
│       ├── main.tf
│       ├── variables.tf
│       └── backend.hcl
└── .github/
    └── workflows/
        └── terraform.yml
```

Every file in this tree is versioned in Git. Every change to production infrastructure goes through a PR. The PR includes the `terraform plan` output as a comment. A human reviews the plan. Only after approval does the plan get applied.

This structure buys you four things that ClickOps fundamentally cannot provide:

**Version history.** `git log infra/environments/production/` tells you every change made to production infrastructure, who made it, when, and why (if they wrote a good commit message). The console has no equivalent. AWS CloudTrail shows API calls, but it does not show you the intent ("changed instance type from t3.medium to t3.large to handle the Q4 traffic spike") or link them to a ticket.

**Rollback.** If the last three commits to production infrastructure broke something, you can `git revert` those commits and open a PR. The plan will show you exactly what will be restored. You can roll back infrastructure the same way you roll back application code. With ClickOps, "rollback" means someone manually re-clicking everything — which they cannot do reliably because they do not know what state they are rolling back to.

**Review.** A second set of eyes on the Terraform plan catches the "1 to destroy" before it becomes a production incident. It also catches security group rules that are too permissive, instance types that are too expensive, and configurations that diverge from the team's standards. Code review for infrastructure is not overhead — it is the control that makes IaC safe at scale.

**Testing in CI.** `terraform validate` checks syntax. `terraform plan` catches provider errors and state conflicts. Tools like [Checkov](https://www.checkov.io/) and [tfsec](https://aquasecurity.github.io/tfsec) run static analysis against the HCL to catch common security misconfigurations (public S3 buckets, security groups open to 0.0.0.0/0, unencrypted EBS volumes). You can run these in CI the same way you run unit tests.

### Tagging and the audit trail

One of the most overlooked benefits of IaC is consistent resource tagging. In a ClickOps environment, tagging is an afterthought — or not a thought at all. The result is an AWS account where nobody can tell which resources belong to which service, which resources are safe to delete, or who to contact when a resource is causing a cost anomaly.

With IaC, tagging is enforced at the provider level. Terraform's `default_tags` block (for the AWS provider) applies a set of standard tags to every resource without requiring the module author to remember to add them:

```hcl
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      ManagedBy   = "terraform"
      Environment = var.environment
      Repository  = "github.com/myorg/infra"
      CostCenter  = var.cost_center
      Team        = var.team
      LastApplied = timestamp()
    }
  }
}
```

The `ManagedBy = "terraform"` tag is particularly useful: it is the signal for AWS Config rules and automation that "this resource should not be modified through the console." You can pair it with an AWS Config rule that alerts whenever a Terraform-managed resource is modified through the console without a corresponding Terraform state change.

Every resource in your AWS account now tells you: who manages it, which environment it belongs to, which team owns it, and which Git repository contains the source of truth. This single change — systematic tagging via IaC — transforms AWS Cost Explorer from an indecipherable wall of resource IDs into an actionable cost breakdown by team, environment, and service.

### Module reuse: the library of standard patterns

Once you have written a VPC module or an EC2 cluster module, you never need to write it again. The Terraform module system allows you to publish reusable infrastructure patterns that any team in your organization can reference, the same way they reference a library function:

```hcl
# infra/environments/staging/main.tf
module "vpc" {
  source = "../../modules/vpc"

  name               = "myapp-staging"
  vpc_cidr           = "10.1.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

module "web_cluster" {
  source = "../../modules/ec2-cluster"

  name             = "myapp-staging-web"
  vpc_id           = module.vpc.vpc_id
  subnet_ids       = module.vpc.public_subnet_ids
  instance_type    = "t3.small"  # Smaller than prod
  desired_capacity = 2
  max_size         = 4
  min_size         = 1
  app_version      = var.app_version
}
```

```hcl
# infra/environments/production/main.tf
module "vpc" {
  source = "../../modules/vpc"

  name               = "myapp-production"
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

module "web_cluster" {
  source = "../../modules/ec2-cluster"

  name             = "myapp-production-web"
  vpc_id           = module.vpc.vpc_id
  subnet_ids       = module.vpc.public_subnet_ids
  instance_type    = "t3.large"  # Production sizing
  desired_capacity = 6
  max_size         = 20
  min_size         = 3
  app_version      = var.app_version
}
```

The staging and production environments are now provably identical in structure. The only differences are in the variable values: instance types, capacities, CIDR ranges. The networking topology, the security group rules, the IAM roles, the tagging — all of it is the same module code, verified once, reused everywhere.

This is the infrastructure equivalent of "build once, promote everywhere." You write the module once, test it in staging, and promote the same module (with different variable values) to production. You are not rebuilding the infrastructure from scratch for each environment — you are parameterizing a single verified template.

The public Terraform Registry at [registry.terraform.io](https://registry.terraform.io) provides community-maintained modules for almost every common infrastructure pattern: VPCs, EKS clusters, RDS instances, Lambda functions, and hundreds more. These modules encode community best practices and save significant time. Verify them before use — the registry is not curated for security — but they are a legitimate starting point.

### The remote state backend

One thing that makes IaC safe to run in a team is the **remote state backend**. By default, Terraform writes `terraform.tfstate` locally. This works for a single engineer experimenting; it fails catastrophically for a team.

The problems with local state:
- Two engineers running `terraform apply` simultaneously will corrupt each other's state.
- The state file contains sensitive values (database passwords, private IPs). Committing it to Git is a security incident.
- If the state file is lost (deleted, corrupted), Terraform loses all knowledge of what it created and will try to create everything again.

The solution is a remote backend with state locking:

```hcl
# infra/environments/production/backend.hcl
# (provided to terraform init -backend-config=backend.hcl)
bucket         = "myapp-terraform-state-prod"
key            = "production/terraform.tfstate"
region         = "us-east-1"
encrypt        = true
dynamodb_table = "terraform-state-lock"
```

```hcl
# infra/environments/production/main.tf
terraform {
  required_version = ">= 1.8.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {}  # config loaded from backend.hcl at init time
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      ManagedBy   = "terraform"
      Repository  = "github.com/myorg/infra"
    }
  }
}
```

The S3 backend with DynamoDB locking gives you:
- **Centralized state** accessible to any CI runner or team member.
- **State locking**: DynamoDB ensures only one `apply` runs at a time; concurrent applies fail fast.
- **Encryption at rest**: the state file (which may contain secrets) is encrypted in S3.
- **Versioning**: S3 object versioning lets you recover from state corruption.

---

## A complete working example: VPC and EC2 with Terraform

Let us walk through a complete, working Terraform configuration for a simple but realistic VPC with a web server. This is the kind of thing a team replaces ClickOps with in the first week of adopting IaC.

```hcl
# infra/modules/vpc/main.tf
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.name}-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.name}-igw"
  }
}

resource "aws_subnet" "public" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone = var.availability_zones[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "${var.name}-public-${var.availability_zones[count.index]}"
    Tier = "public"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.name}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}
```

```hcl
# infra/modules/vpc/variables.tf
variable "name" {
  description = "Name prefix for all resources"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of AZs to create subnets in"
  type        = list(string)
}
```

```hcl
# infra/modules/vpc/outputs.tf
output "vpc_id" {
  description = "The ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = aws_subnet.public[*].id
}
```

```hcl
# infra/modules/ec2-cluster/main.tf
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "web" {
  name        = "${var.name}-web-sg"
  description = "Security group for web servers"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP from anywhere"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS from anywhere"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = {
    Name = "${var.name}-web-sg"
  }
}

resource "aws_launch_template" "web" {
  name_prefix   = "${var.name}-web-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.instance_type

  vpc_security_group_ids = [aws_security_group.web.id]

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    app_version = var.app_version
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "${var.name}-web"
      Version = var.app_version
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "web" {
  name                = "${var.name}-web-asg"
  desired_capacity    = var.desired_capacity
  max_size            = var.max_size
  min_size            = var.min_size
  vpc_zone_identifier = var.subnet_ids

  launch_template {
    id      = aws_launch_template.web.id
    version = "$Latest"
  }

  health_check_type         = "ELB"
  health_check_grace_period = 300

  tag {
    key                 = "Name"
    value               = "${var.name}-web"
    propagate_at_launch = true
  }
}
```

This is real, runnable Terraform. Drop it into `infra/modules/`, reference it from `infra/environments/staging/main.tf`, run `terraform init && terraform plan`, and it will show you exactly what it intends to create. No surprises.

#### Worked example: comparing lead times before and after IaC

A mid-size SaaS company needed a new staging environment for a compliance test. Here is a real-world timeline comparison (illustrative numbers based on common industry patterns):

**Before IaC (ClickOps):**

| Step | Time |
|------|------|
| Write Jira ticket requesting new environment | 30 minutes |
| Wait in ops queue | 5 business days (40 hours) |
| Ops engineer provisions manually (EC2, RDS, ElastiCache, ALB, security groups) | 6 hours |
| Configuration mistakes discovered during testing | 4 hours of back-and-forth |
| Environment ready for testing | **~50 hours elapsed** |
| Environment resembles production? | Unknown — no formal spec |

**After IaC (Terraform + CI):**

| Step | Time |
|------|------|
| Copy `production/` to `staging/`, change `var.environment = "staging"` | 10 minutes |
| Open PR, CI runs `terraform plan` | 8 minutes |
| Reviewer approves PR | 15 minutes |
| Merge, `terraform apply` runs automatically | 45 minutes |
| Environment ready | **~80 minutes elapsed** |
| Environment resembles production? | Yes — same modules, same config, different variable values |

Lead time: **50 hours → 80 minutes**. That is a 37x reduction. The compliance test that previously required a two-week lead time to schedule now takes less time to provision than a long lunch.

The mathematical underpinning of this improvement is straightforward. The ClickOps lead time is dominated by queueing delay (the 5-day wait) and manual execution (the 6-hour session). The IaC lead time is dominated by machine execution (the 45-minute apply) plus a small human review overhead. The human review overhead is actually valuable — it is the safety gate. The queueing delay and manual execution overhead are pure waste.

---

## Drift: the silent enemy of declarative infrastructure

Drift is what happens when the actual state of your infrastructure diverges from the desired state described in your Terraform configuration. It is the IaC equivalent of the snowflake problem, except now it occurs incrementally: your infrastructure starts in a known good state (after a `terraform apply`), and then gradually diverges because someone makes a change outside of Terraform.

![Drift detection catches console-clicked changes that break future plans, while re-apply reconciles infrastructure back to desired state](/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-8.png)

### How drift happens

The most common cause of drift is the "emergency console click." An on-call engineer gets paged at 3 AM because a security group is blocking a health check. They open the console, add the rule, the health check passes, the incident ends. Nobody updates the Terraform configuration. Now the desired state (HCL files) says the security group has rules A, B, and C. The actual state has rules A, B, C, and D (the emergency rule). Terraform does not know about rule D. The next `terraform plan` will show that rule D exists in real infrastructure but is not in the configuration, and will propose destroying it.

The destructive outcome: the next `terraform apply` — run by someone else, days later, who does not know about the emergency change — deletes rule D. The health check breaks again. An incident is created. Hours are spent debugging. The root cause is the original console change that was not codified in Terraform.

Other causes of drift:
- **AWS service updates**: AWS occasionally changes the default value of a parameter. Your resource was created with the old default; the new default shows up as a diff.
- **Out-of-band automation**: a Lambda function or scheduled task modifies infrastructure that Terraform also manages.
- **Terraform state import errors**: a resource was imported with incorrect state, and the real infrastructure never matched what the state file claimed.

### Detecting drift

Terraform detects drift automatically during `terraform plan` through the **refresh** step. When you run `terraform plan`, Terraform reads the current state from the provider APIs (AWS, GCP, etc.) and compares it to the last known state in `terraform.tfstate`. Any differences between "what the state file says should exist" and "what actually exists" are reported as drift.

You can also run `terraform refresh` explicitly to update the state file with the current real-world state (without changing any infrastructure). This is useful before a plan to get a clean view of what drift exists.

For automated drift detection, the pattern is to run `terraform plan` on a schedule in CI and alert if the plan shows any changes:

```yaml
# .github/workflows/drift-detection.yml
name: Terraform Drift Detection

on:
  schedule:
    - cron: "0 9 * * 1-5"  # Every weekday at 9 AM UTC
  workflow_dispatch:

jobs:
  detect-drift:
    name: Detect Infrastructure Drift
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
      issues: write

    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-readonly
          aws-region: us-east-1

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.8.0"

      - name: Terraform Init
        run: terraform init -backend-config=backend.hcl
        working-directory: infra/environments/production/

      - name: Terraform Plan (Drift Check)
        id: plan
        run: |
          terraform plan -detailed-exitcode -no-color 2>&1 | tee drift.txt
          echo "exitcode=${PIPESTATUS[0]}" >> $GITHUB_OUTPUT
        working-directory: infra/environments/production/
        continue-on-error: true

      - name: Alert on Drift
        if: steps.plan.outputs.exitcode == '2'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const drift = fs.readFileSync('infra/environments/production/drift.txt', 'utf8');
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'Infrastructure drift detected in production',
              body: '## Drift Detected\n\n```\n' + drift.slice(-8000) + '\n```\n\n' +
                    'Please investigate and reconcile with `terraform apply`.',
              labels: ['infrastructure', 'drift', 'urgent']
            });
```

The `-detailed-exitcode` flag makes `terraform plan` return exit code 2 if it finds changes to make (drift), exit code 0 if there is nothing to change (clean), and exit code 1 if there is an error. This makes it easy to gate on drift in CI.

### Remediation: what to do when you find drift

When drift detection fires, you have three options depending on the nature of the drift:

**Option 1 — Codify the change in Terraform (most common).** If the drifted change is intentional — the on-call engineer who added the security group rule was right to do it — you update the `.tf` files to include the change, open a PR, and run `terraform apply`. After apply, the desired state matches the actual state. Drift resolved.

**Option 2 — Revert the change via `terraform apply`.** If the drifted change is wrong — someone added a security group rule that should not be there, or changed an instance type without authorization — you run `terraform apply` to revert the infrastructure back to the desired state. This is the "re-apply to reconcile" pattern. The plan will show what is being reverted; the reviewer confirms it is correct.

**Option 3 — Import the changed resource into Terraform state.** If a resource was created outside of Terraform and you want to bring it under Terraform management without recreating it, use `terraform import`:

```bash
# Import an existing EC2 instance into Terraform state
terraform import aws_instance.web i-1234567890abcdef0

# Import an existing security group
terraform import aws_security_group.db sg-0abc123def456
```

After import, run `terraform plan` to verify the imported state matches your HCL definition. If there are diffs, update the HCL to match the real resource, then run another plan. The goal is a plan that shows zero changes — which means your HCL is now the authoritative description of what actually exists.

### The rule: no manual console changes to Terraform-managed resources

Once you adopt IaC, the rule is absolute and admits no exceptions: **you do not make manual changes to infrastructure that Terraform manages.** This is not a soft guideline. It is an architectural invariant. If you need to make an emergency change and cannot wait for a PR, you make the change AND immediately open a PR to codify that change in Terraform. The goal is that the next `terraform plan` shows no unexpected diffs.

This rule sounds harsh, but it is the only way IaC delivers its core promise: that you can recreate your infrastructure from the Git repository alone. The moment console changes are allowed, that promise is broken. The Git repository is no longer the source of truth — it is one of multiple sources of truth, which is the same as no source of truth.

Enforcing this rule technically is more reliable than relying on human discipline alone. The combination of:
1. `ManagedBy = terraform` tags on all resources
2. AWS Config rules that alert on tag violations
3. IAM policies that restrict direct console modifications to specific resources
4. Service Control Policies (SCPs) at the AWS Organization level that deny `ec2:AuthorizeSecurityGroupIngress` for resources tagged `ManagedBy = terraform` unless the caller is the Terraform CI role

...creates a layered enforcement that makes the console-click-after-apply significantly harder to do accidentally. At the organization level, SCPs are the strongest control: they prevent the action entirely, rather than detecting it after the fact.

---

## The IaC delivery pipeline

The full IaC delivery lifecycle mirrors the application code delivery lifecycle closely enough that you can think of them as the same pipeline applied to different kinds of artifacts.

![End-to-end new environment provisioning from writing HCL through PR through CI plan through merge and apply to live infrastructure in under two hours](/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-6.png)

### The full workflow

**Step 1 — Write HCL.** An engineer creates or modifies `.tf` files in the infra repository. This is the "code" half of IaC. For a new environment, this might be as simple as copying an existing environment directory and changing a few variable values. For a new module, it involves writing the resource definitions.

**Step 2 — Local plan.** Before pushing, the engineer runs `terraform plan` locally to see what changes will be made. This catches obvious errors — typos in resource names, missing required variables, syntax errors — before they become CI failures.

**Step 3 — Pull request.** The engineer pushes the branch and opens a PR. The PR triggers the CI pipeline, which runs `terraform fmt -check` (formatting gate), `terraform validate` (syntax gate), static analysis (Checkov, tfsec), and `terraform plan` (diff gate). The plan output is posted as a PR comment.

**Step 4 — Review.** A second engineer reviews both the HCL changes and the plan output. The review focuses on: Is the plan what we expected? Are there any destroys? Are the resource names and tags correct? Do the security group rules follow the team's standards? Is the instance type appropriate for the workload?

**Step 5 — Merge.** The PR is approved and merged to main.

**Step 6 — Apply.** The CI pipeline detects a push to main in the `infra/` directory and triggers `terraform apply`. Because the plan was generated at PR time, the apply should produce identical results to the reviewed plan — unless the underlying AWS state changed between plan and apply (which is rare but possible, and is why the apply stage runs its own plan before applying).

**Step 7 — Monitor.** After apply, drift detection runs on a schedule to catch any out-of-band changes. Alerts go to a Slack channel or create GitHub issues.

### Policy-as-code: gatekeeping the plan

Running `terraform plan` in CI and posting the output to a PR is necessary but not sufficient at scale. When you have multiple teams writing infrastructure code, you need automated policy enforcement that catches violations before a human reviewer has to remember to look for them. This is where **policy-as-code** tools come in.

The three most common options in the Terraform ecosystem are:

**Checkov** — a static analysis tool that scans your HCL files before a plan is even run. It checks for ~1,000 known misconfigurations: public S3 buckets, security groups with overly broad ingress rules, RDS instances with no deletion protection, EC2 instances without IMDSv2 enforcement, and so on. Checkov runs fast (seconds) and fails the CI pipeline before any AWS API calls are made.

**tfsec** — similar to Checkov, focused on security findings. Slightly different rule set; many teams run both.

**OPA/Conftest** — a policy engine that lets you write custom policies in Rego (OPA's policy language). This is more powerful than Checkov for organization-specific rules: "all production RDS instances must have `multi_az = true`", "all public-facing load balancers must have a WAF attached", "instance types in production must be in this approved list." Conftest evaluates the Terraform plan JSON against your Rego policies.

Here is a simple Conftest policy that enforces required tags:

```hcl
# policy/required_tags.rego
package main

required_tags = {"Environment", "Team", "ManagedBy", "CostCenter"}

deny[msg] {
  resource := input.resource_changes[_]
  resource.change.actions[_] == "create"
  tags := resource.change.after.tags
  missing := required_tags - {tag | tags[tag]}
  count(missing) > 0
  msg := sprintf(
    "Resource %s is missing required tags: %v",
    [resource.address, missing]
  )
}
```

You can integrate Conftest into the GitHub Actions workflow:

```yaml
      - name: Terraform Plan (JSON output for policy)
        run: terraform plan -out=tfplan && terraform show -json tfplan > tfplan.json
        working-directory: infra/

      - name: Run OPA Policy Check
        run: |
          conftest test tfplan.json --policy policy/ --all-namespaces
        working-directory: infra/
```

The complete CI gate for a mature IaC pipeline looks like this:

| Gate | Tool | Blocks merge? | Speed |
|------|------|--------------|-------|
| Formatting | `terraform fmt -check` | Yes | ~1 second |
| Syntax validation | `terraform validate` | Yes | ~5 seconds |
| Static security analysis | Checkov / tfsec | Yes | ~30 seconds |
| Custom policy enforcement | OPA/Conftest | Yes | ~10 seconds |
| Plan review | `terraform plan` | Human gate | ~2–5 minutes |
| Plan safety check | Automated destroy check | Yes | ~1 second |

The "automated destroy check" deserves mention: it is a simple script that parses the plan JSON and fails CI if more than N resources are marked for destruction:

```bash
#!/usr/bin/env bash
# check-plan-destroys.sh — fail CI if plan destroys more than threshold
THRESHOLD=${1:-5}
DESTROY_COUNT=$(terraform show -json tfplan | jq '[.resource_changes[] | select(.change.actions[] == "delete")] | length')

if [ "$DESTROY_COUNT" -gt "$THRESHOLD" ]; then
  echo "ERROR: Plan wants to destroy $DESTROY_COUNT resources (threshold: $THRESHOLD)"
  echo "Review the plan carefully before proceeding."
  exit 1
fi

echo "Destroy count: $DESTROY_COUNT (threshold: $THRESHOLD) — OK"
```

This script catches the "accidentally refactored a module and now everything is being recreated" class of incidents automatically, without requiring a human to read the plan output carefully.

---

## Worked example: the VPC to production, with real numbers

#### Worked example: provisioning a full staging environment

A team running a SaaS application needed to add a dedicated staging environment for a series of load tests. Before IaC they relied on the ops team to provision environments manually. After migrating to Terraform they ran the following numbers.

**Environment spec:** 1 VPC, 3 public subnets across 3 AZs, 1 application load balancer, 1 auto-scaling group (2 t3.medium instances), 1 RDS PostgreSQL db.t3.medium with 100 GB storage, 1 ElastiCache Redis cache.t3.micro cluster, security groups, IAM roles, CloudWatch log groups.

**ClickOps baseline (18 months ago):**
- Time to provision: 2 weeks (5 days ops queue + 2 days manual work + 3 days of back-and-forth fixing config mistakes)
- Cost to provision: \$4,200 (ops engineer time at loaded cost)
- Reproducibility: Unknown — no formal documentation
- Environment parity with production: 72% (verified post-hoc by comparing configurations)

**IaC current state:**
- Time to provision: 95 minutes (10 minutes writing changes, 5 minutes CI plan, 15 minutes review, 65 minutes apply)
- Cost to provision: \$140 (engineer time for review + CI compute)
- Reproducibility: 100% — same modules produce same resources every time
- Environment parity with production: 100% — production and staging are different workspaces of the same modules

**Lead time reduction:** 2 weeks → 95 minutes = **30x improvement**
**Cost reduction per environment:** \$4,200 → \$140 = **30x cheaper**

The 100% environment parity point is critically important. Bugs that only show up in production because "staging is slightly different" are a major source of escaped defects. When staging and production are the same Terraform modules with different variable values, that class of bugs disappears.

#### Worked example: catching a destructive plan before it fires

A senior engineer was refactoring the database module to support multi-region replication. They moved the `aws_db_subnet_group` resource from the VPC module to the RDS module. The plan output showed:

```bash
  # module.vpc.aws_db_subnet_group.main will be destroyed
  - resource "aws_db_subnet_group" "main" {
      - id   = "myapp-prod-db-subnet-group" -> null
      - name = "myapp-prod-db-subnet-group" -> null
      - subnet_ids = [
          - "subnet-0abc123",
          - "subnet-0def456",
          - "subnet-0ghi789",
        ]
    }

  # module.rds.aws_db_subnet_group.main will be created
  + resource "aws_db_subnet_group" "main" {
      + name       = "myapp-prod-db-subnet-group"
      + subnet_ids = [
          + "subnet-0abc123",
          + "subnet-0def456",
          + "subnet-0ghi789",
        ]
    }

Plan: 1 to add, 0 to change, 1 to destroy.
```

The reviewing engineer noticed the destroy. Even though the new resource would have the same name and subnet IDs, destroying and recreating a subnet group while an RDS instance is associated with it would cause a brief outage. The correct fix was `terraform state mv module.vpc.aws_db_subnet_group.main module.rds.aws_db_subnet_group.main`, which moves the resource in the state file without touching actual infrastructure. After the state move, `terraform plan` showed 0 changes. The plan gate caught this before any damage was done.

Without the plan gate — without someone reading the plan output before applying — this refactor would have caused a production database outage.

---

## IaC comparison: Terraform, Pulumi, Ansible, and shell scripts

Understanding which tool to use requires understanding the trade-offs clearly. Here is an honest comparison.

![IaC tool comparison matrix across idempotency, state tracking, drift detection, and language dimensions](/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-5.png)

| Tool | Model | State | Language | Best for |
|------|-------|-------|----------|----------|
| Terraform | Declarative | tfstate file | HCL | Multi-cloud infra, teams with mixed backgrounds, existing ecosystem |
| OpenTofu | Declarative | tfstate file | HCL | Terraform-compatible, truly open-source |
| Pulumi | Declarative | Pulumi service or S3 | TypeScript, Python, Go, C# | Teams that prefer real languages, complex logic |
| AWS CDK | Declarative | CloudFormation | TypeScript, Python, Java | AWS-only teams, comfortable with CloudFormation quirks |
| Ansible | Imperative | None (stateless) | YAML | Configuration management, OS-level tasks, not infra provisioning |
| Shell scripts | Imperative | None | Bash / Python | One-off automation, not infrastructure management |

**Terraform** is the dominant tool for a reason: it has the largest module registry (Registry.terraform.io), the widest provider support (AWS, GCP, Azure, Kubernetes, Datadog, PagerDuty, and hundreds more), the most mature state management, and the largest community. If you are starting with IaC and have no strong preference, start with Terraform.

**Pulumi** is the better choice if your team is already expert in TypeScript or Python and finds HCL limiting. Real languages mean real loops, real conditionals, real functions, and real type systems. The trade-off is a steeper learning curve and a somewhat smaller ecosystem.

**Ansible** is frequently misused for infrastructure provisioning. It is genuinely excellent for what it was designed for: configuration management of existing servers (installing packages, managing services, deploying application configs). It is a poor tool for creating and managing cloud infrastructure because it has no state model — it cannot tell you "this EC2 instance already exists; skip creation." Use Ansible alongside Terraform (Terraform provisions the EC2; Ansible configures it), not instead of it.

**Shell scripts** should not manage infrastructure at any scale beyond a personal project. They have no state, no idempotency, no diff output, no rollback, and no drift detection. If you are writing a provisioning script that is more than 50 lines, you should be writing Terraform instead.

---

## War story: the snowflake server that took two weeks to reproduce

The following story is illustrative but closely based on a pattern I have seen multiple times. The details are composited from real incidents.

A fintech startup was running its trading platform on a single EC2 instance they called "the production box." Over eighteen months of operation, this server had accumulated:
- A custom kernel module compiled from source to support a low-latency network driver
- A modified version of PostgreSQL with a proprietary extension, installed from a custom apt repository that no longer existed
- A series of `sysctl` settings documented only in a Slack message from fourteen months earlier
- Three versions of Python installed in conflicting locations because different team members had different preferences
- An nginx configuration with several `location` blocks that nobody could fully explain but that everyone was afraid to touch

The server was running fine. Nobody knew exactly why it worked, but it worked.

Then AWS notified the company that the instance's underlying hardware was being retired. They needed to migrate to a new instance within 30 days.

What followed was a two-week project involving:
- Four engineers running `diff -r /etc` between the old server and freshly-provisioned ones
- Reverse-engineering the custom kernel module from a compiled binary (the source was lost)
- Contacting the company that made the PostgreSQL extension (they still existed but no longer supported that version)
- Rebuilding the apt repository mirror from cached `.deb` files on the old server
- Three failed migrations where the new server passed health checks but processed slightly different results for the same market data inputs

The incident cost approximately \$35,000 in engineering time, \$12,000 in delayed feature work, and nearly cost the company a Series A because their lead investor ran a technical due diligence call during the migration and saw "the team cannot tell us how their production system works."

The migration was eventually successful. The team immediately began writing Terraform for all new infrastructure. The "production box" remained a snowflake for another year until a complete rewrite of the platform, because retroactively IaCing a snowflake server is almost as hard as reproducing one.

The lesson is not "IaC is magic." The lesson is: **the moment you cannot describe your infrastructure as code, you have already lost control of it.** The snowflake does not announce itself. It builds quietly, one console click at a time, until the day you need to reproduce it and discover that you cannot.

### The AWS console session that overrode Terraform state

A second story, shorter. A DevOps engineer was debugging a slow RDS connection. They opened the AWS console, found that the RDS instance's parameter group had `max_connections` set to 100 (Terraform-managed value), and changed it to 500 to see if it helped. It did. They closed the tab.

Six weeks later, a new engineer was onboarding. They ran `terraform plan` to verify they understood the infrastructure. The plan showed:

```bash
  # aws_db_parameter_group.main will be updated in-place
  ~ resource "aws_db_parameter_group" "main" {
        id   = "myapp-prod-pg"
        name = "myapp-prod-pg"
      ~ parameter {
          ~ value = "500" -> "100"
            name  = "max_connections"
        }
    }

Plan: 0 to add, 1 to change, 0 to destroy.
```

The new engineer, not knowing about the console change, ran `terraform apply` (they were debugging something unrelated and ran apply with a `-target` flag that accidentally included the parameter group). The `max_connections` was reset to 100. RDS started refusing connections. An incident was created. The on-call engineer eventually traced it back to the `terraform apply` from two hours earlier. The root cause was the undocumented console change from six weeks before.

Total impact: 45-minute database outage, \$8,000 in lost transaction revenue, one very uncomfortable post-mortem. The fix was a team policy: all Terraform-managed resources are tagged `ManagedBy = terraform`, and those tags are enforced by an AWS Config rule that alerts (and eventually denies) any console change to tagged resources.

---

## When NOT to use IaC

IaC is not the right tool for every situation. Part of being a good platform engineer is knowing when the overhead of a proper IaC workflow is not justified by the complexity of the infrastructure.

**Do not use IaC for:**

**A tiny team on a PaaS.** If you are a three-person startup running entirely on Heroku, Render, Railway, or Fly.io, you are paying the PaaS a premium specifically so you do not have to manage infrastructure. Introducing Terraform is taking on complexity that your PaaS is supposed to absorb. Stay on the PaaS until you genuinely hit its limits.

**Not-yet-stable infrastructure.** The early phase of building a new service often involves rapid iteration: trying different database types, different caching layers, different architectural patterns. Terraform is optimized for stable, long-lived infrastructure. If you are spinning up and tearing down experimental configurations every day, IaC introduces friction without delivering its core benefit (reproducibility matters when you need to reproduce something; if you are going to throw it away, reproduction is irrelevant). Use ClickOps to prototype, then write the IaC when the architecture stabilizes.

**Short-lived one-off resources.** A temporary load-testing environment you will destroy in four hours does not need a PR review and a CI pipeline. Use the console or a one-liner CLI command. The overhead of the IaC workflow is not proportional to the lifetime or value of the resource.

**Configuration management of existing servers.** Terraform is designed to create and manage cloud resources (EC2 instances, VPCs, RDS databases), not to configure the software running on those resources. For OS-level configuration, use Ansible, cloud-init user data, or baked AMIs. The two tools compose well; they should not be confused.

**When your team lacks the discipline to never use the console.** This sounds cynical, but it is real. IaC only delivers its benefits if the team commits to using it as the single source of truth. If some team members will continue making console changes (because it is faster, because they are under pressure, because old habits persist), the IaC configuration will drift and the state file will become unreliable. In this case, the answer is not "accept console changes" — it is to invest in tooling that prevents or detects them (AWS Config rules, CloudTrail alerts, policy-as-code with OPA/Conftest). But until that investment is made, IaC gives a false sense of control that is arguably worse than no IaC.

| Situation | IaC recommendation |
|-----------|-------------------|
| Team of 3, fully on PaaS | Skip IaC — PaaS is the IaC |
| Prototype, architecture not stable | ClickOps until stable, then write IaC |
| Short-lived resource, hours or days | CLI or console — do not open a PR |
| Long-lived, production infra, team of 5+ | IaC mandatory |
| Multi-environment (dev/staging/prod) | IaC strongly recommended |
| Compliance requirements, audit trails | IaC required |

### The cost of adopting IaC too early vs too late

Both directions carry real costs that are often underestimated.

**Adopting too early** (on a two-person team, prototyping phase) means spending more time managing Terraform state, writing modules, and maintaining CI pipelines than building the product. The overhead is real: a Terraform configuration for a simple web app is 400–600 lines of HCL, requires a CI pipeline, requires a remote backend, and requires discipline from every engineer. That investment pays off quickly at five or more engineers with stable infrastructure. At two engineers with an architecture that changes every week, it often slows product delivery without meaningfully improving reliability.

**Adopting too late** (at twenty engineers, six months of production, zero IaC) is more painful. Retroactively importing existing infrastructure into Terraform (`terraform import` for each resource, verifying the state matches, writing the HCL to match what was imported) takes weeks. The real cost is the accumulated technical debt: every console change made before adoption is a potential gap in the IaC configuration that only shows up as surprise drift at the worst possible moment.

The pragmatic recommendation: start IaC when you first commit to a cloud environment being long-lived. For most teams, this is when they provision their production environment for the first time. At that moment, the cost of writing Terraform is low (you would have to configure the resources anyway), and the benefit (reproducibility, auditability, drift detection) starts accruing immediately.

---

## Putting it all together: environment provisioning time before and after

The before-after on environment provisioning deserves its own figure because it captures the full economic case for IaC.

![New environment provisioning drops from a 2-week ClickOps ticket queue to a 2-hour automated IaC pipeline with identical results](/imgs/blogs/infrastructure-as-code-from-clickops-to-declarative-4.png)

The economic case for IaC is strongest in three scenarios:

**Scenario 1 — Multiple environments.** If you have development, staging, and production environments that should be identical except for scale and cost, IaC lets you maintain all three with a single set of modules and three sets of variable values. Without IaC, maintaining three identical environments manually is multiplicatively expensive and inevitably fails.

**Scenario 2 — Disaster recovery.** The DORA time-to-restore metric is dramatically affected by IaC. When a region-wide AWS outage destroys your production environment, the question is: how long does it take to bring production up in a different region? With ClickOps: weeks. With IaC: run `terraform workspace new us-west-2 && terraform apply`, and the answer is hours, dominated by data replication lag, not by the time to recreate infrastructure.

**Scenario 3 — Compliance and audit.** Regulated industries require evidence that infrastructure configurations are reviewed, approved, and recorded. ClickOps fails every compliance audit that asks "show me the change management process for production infrastructure changes." Git history + PR review is compliance evidence that regulators accept.

### Measuring the value of IaC over time

The DORA metrics give you four places to look for IaC's impact:

**Lead time for changes** — the most directly affected metric. Environment provisioning time falls from weeks to hours. Configuration changes that previously required ops ticket queues become self-service PRs. Lead time improvements of 10x–30x are common within six months of IaC adoption.

**Deploy frequency** — indirectly affected. When environment provisioning is self-service and takes hours rather than weeks, teams create more environments (feature environments, experiment environments, load-testing environments). More environments mean more deployment targets. Deploy frequency increases as a result.

**Change-failure rate** — affected through the plan review gate. The plan/apply separation catches destructive changes before they fire. Static analysis (Checkov, tfsec) catches security misconfigurations before they reach production. Teams report 30–60% reductions in infrastructure-related incidents after adopting IaC with proper CI gates (illustrative range based on industry patterns, not a specific measured study).

**Time to restore** — affected through reproducibility. When a production environment can be recreated from a Git repository in hours rather than weeks, the blast radius of catastrophic failures (region outages, accidental deletions, ransomware) is bounded to the data recovery time, not the infrastructure reconstruction time.

The practical measurement approach: pick one metric (lead time for a new environment is the easiest to measure) and track it explicitly before and after IaC adoption. Measure the wall-clock time from "decision to create environment" to "environment serving traffic." Take five measurements before and five after. The improvement is typically dramatic enough to justify the investment on the first measurement.

---

## Key takeaways

1. **ClickOps produces snowflakes.** Manual console changes accumulate into unique, unreproducible server configurations. The divergence is gradual and invisible until you need to reproduce the server.

2. **Declarative IaC declares desired state; imperative IaC executes commands.** Only declarative tools like Terraform provide idempotency, state tracking, and drift detection by design.

3. **Always read the plan before you apply.** The `terraform plan` output is a mandatory artifact, not optional output. It is the primary safety mechanism of declarative IaC. Gate your CI pipeline so every PR includes the plan diff and a human reviews it.

4. **Store state remotely with locking.** Local state fails at team scale. Use S3 + DynamoDB (or Terraform Cloud) for every team project. Never commit state files to Git.

5. **Infrastructure belongs in Git.** Every change to production infrastructure goes through a PR. Every PR includes `terraform plan` output. `terraform apply` only runs from CI after merge. No exceptions for "just a quick change."

6. **Drift is the enemy.** Console changes after `terraform apply` create silent drift that corrupts the desired state. Run scheduled drift detection. Alert on any plan that shows unexpected changes. Codify all emergency changes immediately.

7. **IaC shrinks environment lead time by 30x or more.** The ClickOps lead time for a new environment is dominated by queueing delay and manual execution. IaC replaces both with an automated pipeline that takes minutes to plan and hours to apply.

8. **Not everything needs IaC.** Tiny PaaS teams, unstable prototype architectures, and short-lived resources do not benefit from the overhead of a full IaC workflow. Apply IaC to long-lived, production, multi-environment infrastructure.

9. **Drift detection makes IaC operational.** Provisioning infrastructure once with IaC is only the beginning. Drift detection — scheduled `terraform plan` runs that alert on unexpected changes — is what keeps the IaC model honest over months and years.

10. **The plan/apply separation is the principal safety mechanism.** Separating the read-only diff (plan) from the write operation (apply) is what makes IaC safe enough to automate. Never collapse them into `terraform apply -auto-approve` in a production pipeline without a prior plan review step.

---

## Further reading

- [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the full commit→build→test→package→deploy→operate spine and how IaC fits into the overall delivery system.
- [Terraform in practice: state, modules, and workspaces](/blog/software-development/ci-cd/terraform-in-practice-state-modules-and-workspaces) — the next post in this IaC track: remote state backends, module structure, workspace-per-environment vs directory-per-environment, and the module registry.
- [Managing Terraform safely at scale](/blog/software-development/ci-cd/managing-terraform-safely-at-scale) — drift management, policy-as-code with OPA/Conftest/Sentinel, Atlantis vs Terraform Cloud vs DIY CI, and the organizational patterns that keep IaC from collapsing at 100+ engineers.
- [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) — the SRE companion to IaC: how declarative infrastructure interacts with redundancy, availability zones, and the blast-radius containment strategies that make production resilient. IaC provisions the infra; SRE designs what to provision.
- [HashiCorp Terraform Documentation](https://developer.hashicorp.com/terraform/docs) — the canonical reference. The "Terraform Language" and "CLI" sections are the ones you will actually use day-to-day.
- [DORA State of DevOps Report](https://dora.dev/) — the empirical research behind the DORA metrics. The 2023 and 2024 reports both show strong correlation between IaC adoption and elite delivery performance.
