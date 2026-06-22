---
title: "Terraform in practice: state, modules, and workspaces"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master remote state locking, reusable modules, workspace patterns, drift detection, and the read-before-you-apply discipline that keeps production Terraform safe."
tags:
  [
    "ci-cd",
    "devops",
    "terraform",
    "infrastructure-as-code",
    "iac",
    "platform-engineering",
    "cloud-infrastructure",
    "aws",
    "gitops",
    "software-development",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-1.png"
---

It was 11:47 PM on a Tuesday when the alert fired. The on-call engineer opened the Terraform CI log and felt that particular flavor of dread: three lines in the `apply` output, right below the "Plan: 0 to add, 1 to change, 1 to destroy," there it was — `aws_db_instance.primary` scheduled for replacement. Not an update. A destroy followed by a create. The production RDS instance holding six months of transaction data. The CI job had already started. The engineer had maybe 90 seconds before the `destroy` phase hit.

This is not a hypothetical. It is a direct consequence of one specific Terraform trap: changing an immutable identifier field on a resource causes Terraform to silently schedule a forced replacement. The plan output said it clearly, in the middle of 200 lines of diff that nobody had looked at carefully because the pipeline was in a hurry. The engineer killed the job in time. The post-mortem identified three contributing failures: no mandatory human review of the plan, no `-target` safeguard, and a CI pipeline that auto-applied without a gate. All three were process failures. All three had Terraform-specific mitigations that a more experienced team had already learned.

This post is the collection of those lessons. It covers the intermediate mechanics of Terraform — the parts that matter once you have written your first `.tf` file but before you are confidently running it on production infrastructure every day. Remote state and locking, why local state is a liability. Modules and how to structure them for reuse without turning a three-environment deployment into a maintenance nightmare. Workspaces and when directories beat them. Drift detection and the `import` command for IaC-ifying hand-built infrastructure. And the single most important discipline in all of Terraform: reading the plan output before you type `apply`. By the end you will have a VPC-plus-EKS module structure, a multi-environment directory layout, a CI pipeline that plans on PR and applies on merge, and enough battle context to avoid the traps that have destroyed production databases at companies you have heard of.

![The Terraform remote state flow showing how S3 and DynamoDB protect state from concurrent applies](/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-1.png)

## 1. Why state is the most dangerous file in your repository

Every Terraform operator eventually learns a hard truth: the `.tf` files are not the source of truth about your infrastructure. The state file is. The `.tf` files are a *desired* state. The state file is a *map* — a JSON document that records every resource Terraform manages, the arguments it was created with, the provider-assigned identifiers (ARNs, resource IDs, IP addresses), and the dependencies between resources. Without that map, Terraform cannot answer the most basic question: "Does this resource already exist?"

The state file is also a secret vault. It stores every value that passed through a resource argument at creation time, including the values of `password`, `secret_access_key`, `private_key`, and any other sensitive attribute. Terraform 1.x can mark an output as `sensitive = true` to suppress it from CLI output, but it still writes the plaintext value to state. A state file for a modest AWS environment typically contains database passwords, Redis auth tokens, and the initial admin credentials for every managed IAM user. Keep that in mind when you consider where it lives.

### What local state actually means

When you run `terraform init` without a `backend` block, Terraform defaults to the `local` backend. State is written to `terraform.tfstate` in the current directory. This is fine for a tutorial. It is catastrophic for a team.

The failure modes are straightforward. First, the file lives on whoever ran the last `apply`. If that person's laptop is lost, stolen, or wiped, the state file disappears. Terraform no longer knows what it manages. The infrastructure still exists in AWS — but Terraform will try to create duplicates of everything, or refuse to operate until you reconstruct state manually. Second, there is no locking. If two engineers run `terraform apply` at the same time against the same configuration, both processes read the current state, compute a diff, apply their changes, and write back state. Whichever write finishes last wins, silently discarding the other's changes. You now have infrastructure that does not match state, and state that does not match either apply. This is state corruption. Recovering from it manually is one of the most unpleasant days you can have as an infrastructure engineer. Third, the file gets committed to Git. This happens more often than you would expect, because `terraform.tfstate` is not in the default `.gitignore` shipped with most project templates. Once it is in Git history, the secrets inside are accessible to anyone with read access to the repository, including everyone who ever had access. Rotating those secrets after the fact requires finding and rotating every sensitive value that ever appeared in state — a task that often takes days.

![Local state stored on a developer laptop compared to remote state in S3 with DynamoDB locking](/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-3.png)

### The S3 + DynamoDB backend

The standard remote state backend for AWS-hosted infrastructure is S3 for storage and DynamoDB for locking. The S3 bucket holds the state file. DynamoDB provides a distributed lock via a `LockID` attribute — when `terraform plan` or `terraform apply` starts, it attempts to write a lock item to the table. If an item already exists (another run is in progress), the new run fails immediately with a clear error: `Error: Error locking state: Error acquiring the state lock`. No silent corruption. No half-applied concurrent changes.

```hcl
# backend.tf — configure once per root module
terraform {
  backend "s3" {
    bucket         = "acme-terraform-state-prod"
    key            = "clusters/prod-us-east-1/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-east-1:123456789:key/mrk-abc123"
    dynamodb_table = "terraform-state-lock"
  }
}
```

A few details matter here. The `encrypt = true` flag tells S3 to use server-side encryption. Without it, the state file sits in plaintext on S3. Adding `kms_key_id` upgrades to a customer-managed KMS key, which gives you key rotation, access auditing in CloudTrail, and the ability to revoke access by disabling the key. The `dynamodb_table` must have a partition key named exactly `LockID` of type String — Terraform looks for that specific attribute name.

The bucket itself needs versioning enabled. State versioning means that a failed `apply` that wrote a corrupt state does not permanently destroy the previous state — you can roll back to the last known-good version. S3 versioning is the state-equivalent of a database backup, and it has saved engineers from disaster more than once.

```hcl
# bootstrap/state-backend.tf — create the backend resources themselves
# (this root module uses LOCAL state — chicken-and-egg, run once manually)
resource "aws_s3_bucket" "tf_state" {
  bucket = "acme-terraform-state-prod"
}

resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.tf_state.arn
    }
  }
}

resource "aws_dynamodb_table" "tf_lock" {
  name         = "terraform-state-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}
```

One non-obvious issue: you cannot use S3 as a backend before the S3 bucket exists. The bootstrap module that creates the bucket and DynamoDB table must itself run with local state, just once. After that, every other root module can declare the S3 backend and point to it. Some teams use a dedicated "bootstrap" AWS account and Terraform project for this, managed by a small group of infra admins. The rest of the team never touches the backend bucket directly.

### State as sensitive data — access controls matter

Because state contains secrets, access to the state S3 bucket must be treated like access to a secrets manager. Only the CI/CD service role and a small group of infrastructure administrators should have `s3:GetObject` and `s3:PutObject` on the state bucket. Developer IAM roles should have at most `s3:ListBucket` to run plan (plan reads state). The `terraform apply` IAM role should be a separate identity from the `terraform plan` role, with the apply role gated behind a manual approval step or a CI merge gate. This is the IAM equivalent of requiring a second key for a bank vault.

State encryption with a KMS key that only the CI role can use means that even if the S3 bucket ACL is misconfigured, raw access to the bucket bytes does not yield readable secrets without the KMS key. Defense in depth.

## 2. Modules: the unit of reusable infrastructure

A Terraform module is any directory containing `.tf` files. Every root configuration is a module. What most people mean by "module" is a *child module* — a directory that you call from a root module (or another module) via a `module` block. Child modules are how you package, version, and reuse infrastructure patterns.

The motivation is the same as functions in application code: you do not want to copy-paste the same VPC configuration into every environment's root module. You want to define the VPC pattern once, parameterize the parts that vary (CIDR ranges, availability zones, tags), and call it from each environment's root module with different variable values. When you need to change the VPC pattern — adding a new subnet tier for database isolation, for example — you change it in one place and all environments inherit the change on their next `apply`.

### Writing a module: inputs, outputs, locals

A module has three interface surfaces:

1. **Variables** (`variables.tf`) — the inputs. They are declared with `variable` blocks. Callers pass values. Mark sensitive variables with `sensitive = true`.
2. **Outputs** (`outputs.tf`) — the things the module exposes to its caller. Other modules and the root module read these. Mark sensitive outputs with `sensitive = true`.
3. **Locals** (`locals.tf` or inline) — intermediate computed values used inside the module but not exposed to callers. Locals avoid duplication without leaking internal details.

```hcl
# modules/vpc-baseline/variables.tf
variable "vpc_cidr" {
  type        = string
  description = "The CIDR block for the VPC, e.g. 10.0.0.0/16"
}

variable "availability_zones" {
  type        = list(string)
  description = "List of AZs to create subnets in"
}

variable "environment" {
  type        = string
  description = "Environment name — used as a resource tag prefix"
}

variable "enable_nat_gateway" {
  type    = bool
  default = true
}
```

```hcl
# modules/vpc-baseline/locals.tf
locals {
  # Derive subnet CIDRs automatically from the VPC CIDR
  public_subnets  = [for i, az in var.availability_zones :
    cidrsubnet(var.vpc_cidr, 8, i)]
  private_subnets = [for i, az in var.availability_zones :
    cidrsubnet(var.vpc_cidr, 8, i + 10)]
}
```

```hcl
# modules/vpc-baseline/outputs.tf
output "vpc_id" {
  value       = aws_vpc.main.id
  description = "The ID of the VPC"
}

output "private_subnet_ids" {
  value       = aws_subnet.private[*].id
  description = "List of private subnet IDs"
}

output "public_subnet_ids" {
  value       = aws_subnet.public[*].id
  description = "List of public subnet IDs"
}
```

Calling the module from a root module:

```hcl
# environments/prod/main.tf
module "vpc" {
  source = "../../modules/vpc-baseline"

  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  environment        = "prod"
  enable_nat_gateway = true
}

module "eks" {
  source = "../../modules/eks-cluster"

  cluster_name    = "prod-us-east-1"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids
  node_group_size = 3
}
```

Notice how the `eks` module receives `vpc_id` and `subnet_ids` from the `vpc` module's outputs. Terraform builds a dependency graph from these references and runs resource creation in the correct order, in parallel where possible. You never explicitly specify "create VPC before EKS" — the data dependency makes it implicit.

![Terraform module layer stack showing root module delegating to environment and shared modules above the provider API](/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-2.png)

### Module versioning: why `?ref=` matters

Unversioned modules are a footgun. If you reference a module as `source = "../../modules/vpc-baseline"` with a relative path, then every change to the module immediately affects every caller. In a monorepo this is sometimes acceptable. In a shared infra library consumed by multiple teams or multiple repositories, it is not — a VPC change intended for one team silently breaks another team's next `apply`.

The fix is to version modules using Git tags. Publish the module to a Git repository (private GitHub, GitLab, or the public Terraform Registry), tag releases, and reference specific tags in callers:

```hcl
module "vpc" {
  source = "git::https://github.com/acme/terraform-modules.git//modules/vpc-baseline?ref=v2.3.0"

  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  environment        = "prod"
}
```

The double slash `//` separates the repository URL from the subdirectory within the repository. The `?ref=v2.3.0` pins to a specific tag. To upgrade, you change the tag reference, run `terraform init -upgrade` to download the new version, run `plan` to review the changes, and merge only after confirming the diff is expected. This is semantic versioning for infrastructure, and it is exactly as valuable as semantic versioning for application libraries.

For the Terraform public registry (`registry.terraform.io`), the `version` argument does the pinning:

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "prod-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["us-east-1a", "us-east-1b", "us-east-1c"]
}
```

The `~> 5.0` constraint allows patch and minor updates within the 5.x series but blocks upgrades to 6.0. Always pin module versions. "Latest" is a risk you cannot quantify.

### The flat-module anti-pattern

The flat-module anti-pattern occurs when a root module directly instantiates dozens of resource blocks without any child module structure. Everything is in one directory. 3,000 lines of Terraform in a single `main.tf`. Every `plan` takes four minutes because Terraform must refresh the state of every resource. Every `apply` has maximum blast radius — change one variable, and anything that depends on that variable is re-evaluated. Testing is impossible.

The opposite anti-pattern — too many thin modules — is also real. If every single AWS resource gets its own module, you end up with 80 modules each containing two resources. The overhead of module versions, variables, and outputs overwhelms any organizational benefit.

The right granularity is a module per *deployed system concern*: a `vpc-baseline` module, an `eks-cluster` module, an `rds-postgres` module, an `iam-service-account` module. Each encapsulates one logical component and exposes only the outputs that other components need to reference it. Modules that are 200–500 lines of HCL and contain 10–30 resources are generally in the right range. Modules smaller than that are probably not worth the module overhead. Modules larger than that are probably hiding multiple concerns that should be separated.

### Testing a module

A Terraform module is code. Code without tests rots. There are two practical approaches:

**Terratest** (Go-based): write a Go test that calls `terraform.InitAndApply()` against the module with test inputs, then uses the AWS SDK to assert the real infrastructure matches expectations, then calls `terraform.Destroy()` to clean up. This gives you integration-level confidence. It is slow (real AWS resources take minutes to create) and costs real money, but it is the highest-fidelity test you can run.

**`terraform test` (built-in since 1.6)**: a simpler built-in test framework. Write `.tftest.hcl` files in a `tests/` directory alongside the module. Each test file declares `run` blocks that call the module, assert on outputs, and optionally assert on resource attributes using `assert` blocks. It still provisions real infrastructure by default, but can run against mock providers for pure unit testing.

```hcl
# modules/vpc-baseline/tests/basic.tftest.hcl
variables {
  vpc_cidr           = "10.1.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b"]
  environment        = "test"
  enable_nat_gateway = false
}

run "creates_vpc_with_correct_cidr" {
  command = apply

  assert {
    condition     = output.vpc_id != ""
    error_message = "VPC ID should not be empty"
  }
}
```

At minimum, run `terraform validate` and `terraform plan` in CI against a module any time a PR modifies it, using a dedicated test variable file. This catches syntax errors and type mismatches without provisioning real infrastructure.

For modules that are shared across teams or that manage critical infrastructure, the testing maturity progression is:

1. **`terraform validate`**: catches syntax errors and type mismatches. Takes seconds. Free. Run on every commit.
2. **`terraform plan` against a test variable file**: catches reference errors and missing required variables. Takes 30–90 seconds (local provider checks). Run on every PR.
3. **`terraform test` with mock providers**: runs assertions against module outputs without creating real AWS resources. Takes 1–2 minutes. Useful for catching logic errors in locals and output computations.
4. **Terratest integration tests**: provision real AWS resources, run assertions, destroy. Takes 5–20 minutes and costs real money. Run on a schedule (nightly, weekly) or before a major module version bump.

Most teams get 90% of the benefit from the first two levels. The last two matter for modules that provision resources with complex behavior (EKS clusters, RDS with complex parameter groups, network ACLs) where the plan-only check cannot validate that the provisioned resource actually works as expected.

## 3. Workspaces: environment isolation with important caveats

A Terraform workspace is a named instance of state within a single backend. The default workspace is called `default`. You create a new one with `terraform workspace new staging`, switch with `terraform workspace select staging`, and Terraform stores a separate state file for each workspace under the same backend bucket. Running `terraform apply` in the `staging` workspace updates only the staging state file. The production state file is untouched.

This looks like exactly what you want for environment isolation. And it is — for the specific case where two environments are structurally identical and differ only by a small set of variable values (a dev cluster with one node group vs prod with three, for example). Workspaces excel at this.

The problem surfaces when environments drift. Production adds a WAF, a second load balancer, and a CloudFront distribution that staging never had. Now the root module has `count = var.environment == "prod" ? 1 : 0` conditionals sprinkled throughout. The `plan` output for staging is polluted with resources that only prod has. Reviewing a change becomes an exercise in mentally filtering environment-specific noise. After 18 months, the root module has 20 environment-specific conditionals and nobody is confident they understand all the interactions.

The workspace anti-pattern is using workspaces for environments that have meaningfully different infrastructure topology. The workspace model assumes one root module works for all workspaces. When that assumption breaks, workspaces hide the divergence behind workspace names rather than making it explicit in code.

![Matrix comparing workspace-per-environment against directory-per-environment on drift isolation, PR plan testing, state separation, and complexity](/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-4.png)

### Directories with shared modules: the more explicit pattern

The alternative is a directory per environment, each with its own `main.tf`, each calling shared modules but with entirely independent variable files and state:

```bash
terraform/
├── modules/
│   ├── vpc-baseline/
│   ├── eks-cluster/
│   └── rds-postgres/
├── environments/
│   ├── dev/
│   │   ├── main.tf        # calls modules with dev-sized inputs
│   │   ├── variables.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf     # state key: environments/dev/terraform.tfstate
│   ├── staging/
│   │   ├── main.tf        # staging topology — may differ from dev
│   │   ├── variables.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf     # state key: environments/staging/terraform.tfstate
│   └── prod/
│       ├── main.tf        # prod topology — WAF, multi-AZ, higher limits
│       ├── variables.tf
│       ├── terraform.tfvars
│       └── backend.tf     # state key: environments/prod/terraform.tfstate
```

Each environment directory is a separate root module with its own state file. Prod can have resources that staging never has, explicitly, without any `count = var.environment == "prod"` tricks. A CI job can run `terraform plan` against each directory independently on a PR, producing per-environment plan outputs that reviewers check separately. State is isolated by default. The cost is more files to maintain and explicit module version bumps across all three environments (which is a feature, not a bug — you can upgrade dev first, validate, then upgrade staging and prod separately).

The decision rule is simple:

| If environments are... | Use... |
|---|---|
| Structurally identical (same resources, different sizes) | Workspaces — less overhead, fine |
| Topologically different (prod has resources dev lacks) | Directories — explicit is better |
| Managed by different teams or different cadences | Directories — independent state |
| Sharing 90%+ of modules, differing only in var values | Workspaces — acceptable |

When in doubt, use directories. The explicitness always wins over time.

## 4. Drift: the gap between desired and actual

Drift is when the real-world infrastructure diverges from what Terraform state records. It happens in three ways. A human logs into the AWS console and changes a security group rule — call it "emergency clickops." A non-Terraform process modifies a resource — an auto-scaling group changes the instance count, a Lambda function's concurrency limit is changed by a CloudWatch alarm. Or Terraform itself has a bug that misrepresents a resource's actual state.

Drift is dangerous because the next `terraform plan` will show the drift as a change-to-be-applied, potentially reverting a deliberate emergency fix. If the on-call engineer widened a security group rule at 2 AM to allow a new monitoring agent to connect, and then CI runs a scheduled `plan` at 3 AM, the plan output will contain a line that removes that rule. If `apply` runs automatically, the fix is silently reverted.

### Detecting drift

`terraform plan` is already a drift detector — any discrepancy between state and real-world shows up as a proposed change. Running `terraform plan` on a schedule (nightly or hourly) against each environment and alerting on any non-empty plan output is a basic drift detection system. Most teams also run `terraform plan` in CI on merge to `main`, so any drift is caught before the next intentional change is applied.

`terraform refresh` updates the state file to match actual AWS resource state without changing any resources. This is useful when you know drift has occurred and you want state to reflect it before planning. Note that `refresh` is implicit in `plan` by default unless you pass `-refresh=false`.

The Atlantis and Spacelift platforms extend this by running automatic speculative plans on every PR and posting the diff as a PR comment, letting reviewers see exactly what infrastructure will change before they merge.

### Importing existing resources

The hardest version of the drift problem is the one every infrastructure team eventually faces: you have hand-built infrastructure — VPCs, RDS clusters, S3 buckets created through the console over the past two years — and you need to bring it under Terraform management. The resources exist in AWS but not in any Terraform state file.

The classic approach is `terraform import`. You write a resource block in your `.tf` files that matches the existing resource's configuration, then run `import` to associate the real resource's ID with the Terraform resource address:

```bash
# Import an existing RDS instance into Terraform state
terraform import aws_db_instance.primary db-prod-abc12345

# Import an existing S3 bucket
terraform import aws_s3_bucket.artifacts my-existing-bucket-name

# Import an existing EKS cluster
terraform import aws_eks_cluster.main prod-us-east-1
```

After import, you run `terraform plan`. If your resource block accurately describes the real resource, the plan shows no changes. If the plan shows differences, you must update your `.tf` file to match reality — attributes that differ will be applied (and may cause unwanted changes). The goal is to reach a zero-diff plan before you allow CI to run `apply`.

Terraform 1.5 introduced `import` blocks, a declarative version that belongs in `.tf` files rather than being a CLI-only operation:

```hcl
# imports.tf — declarative import, reviewed in PRs like any other change
import {
  to = aws_db_instance.primary
  id = "db-prod-abc12345"
}

import {
  to = aws_s3_bucket.artifacts
  id = "my-existing-bucket-name"
}
```

Terraform 1.5 also added `terraform plan -generate-config-out=generated.tf`, which auto-generates the resource block definitions from the real AWS resource attributes. The generated code is ugly but complete — it gives you a starting point that you can clean up, rather than requiring you to write the entire resource block from memory.

The import workflow is:

```bash
# Step 1: generate resource block definitions from real infra
terraform plan -generate-config-out=generated.tf

# Step 2: review and clean up generated.tf
# Step 3: run plan to see if any diffs remain
terraform plan

# Step 4: if plan is clean, commit generated.tf and run apply
terraform apply
```

One caution: import operations and the subsequent `apply` are operations on real production infrastructure. Run them during a maintenance window or with careful `-target` scoping to limit the blast radius to the specific resources you are importing.

The common failure mode in import projects is scope creep. You start with the intent to import the production VPC. You run `terraform plan -generate-config-out=generated.tf`. The generated file has 87 resources because the VPC is associated with route tables, subnets, security groups, NAT gateways, and flow logs. Each of those 87 resources needs an `import` block. Each needs its AWS resource ID, which requires looking it up in the console or AWS CLI. By the time you have a zero-diff plan, three weeks have passed. The lesson: import in small batches, one logical component at a time. Import the VPC and its core subnets first, validate, commit. Then import the security groups. Then the NAT gateways. Breaking the import into a series of small PRs, each with a zero-diff plan, makes the process auditable and recoverable if something goes wrong.

## 5. The read-before-you-apply discipline

This is the principle that matters most. More than any tooling choice, more than any workflow optimization, the single practice that prevents production incidents in Terraform is: **always read the plan output before you type `apply`, and read it completely**.

The plan output is Terraform's promise about what it will do to your infrastructure. Every resource in the plan is annotated with one of three symbols:

- `+` — will be **created** (new resource, did not exist before)
- `-` — will be **destroyed** (resource exists, will be deleted)
- `~` — will be **updated in-place** (resource exists, attributes will be changed without replacement)
- `-/+` — will be **destroyed then recreated** (forced replacement; semantically equivalent to destroy followed by create)

The `~` symbol is generally safe. A security group rule update, a tag change, an instance type change that AWS applies in place — these modify the resource without deleting and recreating it. The `-/+` symbol is the one that kills databases.

![Safe in-place update compared to the dangerous forced replacement trap where changing an immutable field triggers destroy then recreate](/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-6.png)

### The forced-replacement trap

Not all resource attributes can be updated in place. Some attributes are "immutable" — once set at creation time, they can never be changed. Changing them requires destroying and recreating the resource. AWS imposes this constraint at the API level. When Terraform detects that you have changed an immutable attribute, it marks the resource `-/+` and schedules the destroy-then-recreate.

Common immutable attributes:

- `aws_db_instance`: `identifier`, `db_subnet_group_name`, `engine`, `username`
- `aws_eks_cluster`: `name`, `role_arn`
- `aws_elasticache_cluster`: `cluster_id`, `subnet_group_name`
- `aws_lb` (ALB): `name`, `internal`, `load_balancer_type`
- `aws_s3_bucket`: `bucket` name

The pattern that causes most incidents: a new team member writes infrastructure code and names a resource to match a new naming convention. "Our convention is `{service}-{env}-{region}`, but this RDS was created as `prod-db` — let me fix that." They change the `identifier` field in the Terraform file. The plan shows `-/+` for `aws_db_instance.primary`. If nobody reads the plan carefully, the apply destroys the production database.

The mitigation is a combination of CI enforcement and human review. In CI, post the plan output as a pull request comment (Atlantis does this automatically; you can also run `terraform plan -out=plan.tfplan && terraform show -json plan.tfplan > plan.json` and parse the JSON for dangerous operations). Require a manual approval step in the CI pipeline before `apply` runs if the plan contains any `-` or `-/+` operations. Do not auto-apply plans that include destroys.

```bash
# Check if plan contains any destroys — fail CI if so, require human review
terraform plan -out=tfplan
terraform show -json tfplan | jq '.resource_changes[] | select(.change.actions | contains(["delete"])) | .address'
```

If the above `jq` query produces any output, the plan destroys at least one resource. Gate on this in CI.

### The `-target` escape hatch

Sometimes you need to apply a change to a specific resource without touching everything else in the plan. The `-target` flag scopes `plan` and `apply` to a specific resource address:

```bash
# Apply only the security group change, leaving the rest of the plan untouched
terraform apply -target=aws_security_group.eks_nodes

# Plan only the RDS module
terraform plan -target=module.rds
```

Use `-target` as an emergency surgical tool, not as a workflow pattern. Repeated use of `-target` creates state drift — resources that should be in sync are now partially applied. Every `-target` apply should be followed, eventually, by a full `apply` to reconcile the entire state. Some teams prohibit `-target` in CI pipelines entirely for exactly this reason.

## 6. A complete VPC + EKS module structure

Here is a concrete module structure for a production-grade VPC and EKS cluster, organized for multi-environment use.

```bash
terraform/
├── modules/
│   ├── vpc-baseline/
│   │   ├── main.tf        # VPC, subnets, IGW, NAT, route tables
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── locals.tf
│   │   └── tests/
│   │       └── basic.tftest.hcl
│   ├── eks-cluster/
│   │   ├── main.tf        # EKS cluster, node groups, IAM roles
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── tests/
│   │       └── basic.tftest.hcl
│   └── rds-postgres/
│       ├── main.tf        # RDS instance, subnet group, param group
│       ├── variables.tf
│       └── outputs.tf
├── environments/
│   ├── dev/
│   │   ├── backend.tf
│   │   ├── main.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   │   ├── backend.tf
│   │   ├── main.tf
│   │   └── terraform.tfvars
│   └── prod/
│       ├── backend.tf
│       ├── main.tf
│       └── terraform.tfvars
└── bootstrap/            # Creates the S3 backend — run once manually
    ├── main.tf
    └── state-backend.tf
```

The `environments/prod/main.tf` calls the shared modules:

```hcl
# environments/prod/main.tf
locals {
  env    = "prod"
  region = "us-east-1"
}

module "vpc" {
  source = "../../modules/vpc-baseline"

  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  environment        = local.env
  enable_nat_gateway = true
}

module "eks" {
  source = "../../modules/eks-cluster"

  cluster_name           = "acme-${local.env}-${local.region}"
  kubernetes_version     = "1.30"
  vpc_id                 = module.vpc.vpc_id
  subnet_ids             = module.vpc.private_subnet_ids
  node_group_min_size    = 3
  node_group_max_size    = 12
  node_group_desired     = 6
  node_instance_type     = "m5.2xlarge"
  environment            = local.env
}

module "rds" {
  source = "../../modules/rds-postgres"

  identifier         = "acme-${local.env}-pg"
  engine_version     = "16.2"
  instance_class     = "db.r6g.xlarge"
  allocated_storage  = 500
  multi_az           = true
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  environment        = local.env
}
```

```hcl
# environments/prod/terraform.tfvars
vpc_cidr           = "10.0.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
```

```hcl
# environments/dev/terraform.tfvars
vpc_cidr           = "10.1.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b"]
```

Each environment directory is independent. Dev uses two AZs and a smaller CIDR. Prod uses three AZs, multi-AZ RDS, and larger instances. The modules are the same; the variable inputs differ.

#### Worked example: measuring the module refactor payoff

Before the module refactor, a five-engineer platform team maintained three copy-pasted VPC configurations across dev, staging, and prod. When they needed to add a new private subnet tier for database isolation, they made the change in prod first and forgot to propagate it to staging. Six weeks later, a developer tried to deploy a database-backed service to staging and hit a routing error because the DB subnet group did not exist. Root cause: configuration drift between environments. Time to diagnose and fix: four hours.

After the module refactor:

- A change to `modules/vpc-baseline` is reviewed once in a single PR
- CI runs `terraform plan` against all three environment directories when any module file changes (via GitHub Actions path filters `paths: ['terraform/modules/vpc-baseline/**']`)
- The dev plan is reviewed, applied, and validated before the staging and prod PRs are opened
- Time to propagate a subnet change across all three environments: 40 minutes, not four hours
- Configuration drift incidents: zero in the 12 months after the refactor (illustrative, based on the typical outcome of this kind of consolidation)

The DORA lead-time metric improved as well. Before: a new database subnet required a separate Jira ticket per environment, manual console work, and three separate review cycles. After: one PR, three plan outputs, one review. Lead time from "we need a new subnet tier" to "it exists in all three environments": two days versus the previous three weeks.

## 7. A CI pipeline that plans on PR and applies on merge

The Terraform workflow in CI has two jobs: a plan job that runs on every PR and posts the diff as a comment, and an apply job that runs only on merge to `main`. This is the pattern that keeps humans in the loop for every infrastructure change.

![The complete Terraform CI workflow from init through plan on PR to human review to merge to apply and drift monitoring](/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-5.png)

```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  pull_request:
    paths:
      - 'terraform/**'
  push:
    branches:
      - main
    paths:
      - 'terraform/**'

permissions:
  id-token: write    # OIDC federation to AWS
  contents: read
  pull-requests: write  # post plan as PR comment

jobs:
  plan:
    name: Plan (${{ matrix.environment }})
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    strategy:
      matrix:
        environment: [dev, staging, prod]
    defaults:
      run:
        working-directory: terraform/environments/${{ matrix.environment }}
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}:role/terraform-plan-role
          aws-region: us-east-1

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.9.0"

      - name: Terraform init
        run: terraform init

      - name: Terraform validate
        run: terraform validate

      - name: Terraform plan
        id: plan
        run: |
          terraform plan -out=tfplan -no-color 2>&1 | tee plan-output.txt
          EXIT_CODE=${PIPESTATUS[0]}
          echo "exit_code=$EXIT_CODE" >> $GITHUB_OUTPUT
          exit $EXIT_CODE

      - name: Check for dangerous operations
        run: |
          DESTROYS=$(terraform show -json tfplan | \
            jq '[.resource_changes[] | select(.change.actions | contains(["delete"]))] | length')
          echo "Resources to destroy: $DESTROYS"
          if [ "$DESTROYS" -gt 0 ]; then
            echo "WARNING: plan includes $DESTROYS destroy operations — require explicit approval"
            echo "destroy_count=$DESTROYS" >> $GITHUB_ENV
          fi

      - name: Post plan as PR comment
        uses: actions/github-script@v7
        if: github.event_name == 'pull_request'
        with:
          script: |
            const fs = require('fs');
            const planOutput = fs.readFileSync('terraform/environments/${{ matrix.environment }}/plan-output.txt', 'utf8');
            const destroyCount = '${{ env.destroy_count }}' || '0';
            const warning = parseInt(destroyCount) > 0
              ? `⚠️ **${destroyCount} resource(s) will be DESTROYED**\n\n`
              : '';
            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Terraform Plan — \`${{ matrix.environment }}\`\n\n${warning}\`\`\`\n${planOutput.slice(-65000)}\n\`\`\``
            });

  apply:
    name: Apply (${{ matrix.environment }})
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    environment: ${{ matrix.environment }}  # GitHub environment protection rules gate this
    strategy:
      matrix:
        environment: [dev, staging, prod]
      max-parallel: 1  # apply environments sequentially: dev first, then staging, then prod
    defaults:
      run:
        working-directory: terraform/environments/${{ matrix.environment }}
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}:role/terraform-apply-role
          aws-region: us-east-1

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.9.0"

      - name: Terraform init
        run: terraform init

      - name: Terraform apply
        run: terraform apply -auto-approve -input=false
```

A few design choices in this pipeline deserve explanation. The plan job uses a `terraform-plan-role` with read-only IAM permissions — it can describe resources and read state but cannot create or modify anything. The apply job uses a `terraform-apply-role` with broader permissions. This means that plan jobs can run freely on any branch without risk, while apply is restricted to the main branch and gated by GitHub environment protection rules. The `environment: prod` configuration in the apply job allows you to require a manual approval from a designated reviewer before the prod apply runs, even though the staging apply has already completed.

The `max-parallel: 1` on the apply matrix is intentional: apply environments sequentially (dev → staging → prod), so a problem in staging stops before prod.

Authentication uses OIDC federation (`id-token: write` permission) — the CI job requests a short-lived token from AWS STS rather than storing long-lived AWS access keys as secrets. This is the correct pattern; long-lived CI secrets are a supply-chain risk, as discussed in the series intro post at [/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model).

## 8. Drift monitoring: the scheduled plan

A CI plan-on-PR catches drift before a planned change is applied. But drift can occur between changes — someone makes an emergency console change at 3 AM, and the next planned change might be days away. By then, the drift is deep and the remediation is disruptive.

The solution is a scheduled `terraform plan` that runs against all environments on a regular cadence (nightly is common; some teams run it hourly) and alerts on any non-empty plan output.

```yaml
# .github/workflows/drift-detection.yml
name: Drift Detection

on:
  schedule:
    - cron: '0 6 * * *'  # every day at 06:00 UTC
  workflow_dispatch:       # also allow manual trigger

jobs:
  detect-drift:
    name: Drift check (${{ matrix.environment }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [dev, staging, prod]
    defaults:
      run:
        working-directory: terraform/environments/${{ matrix.environment }}
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}:role/terraform-plan-role
          aws-region: us-east-1

      - uses: hashicorp/setup-terraform@v3

      - name: Terraform init
        run: terraform init

      - name: Check for drift
        id: drift
        run: |
          terraform plan -detailed-exitcode -no-color 2>&1 | tee plan-output.txt
          EXIT=$?
          # exit code 0 = no changes, 1 = error, 2 = changes present
          echo "exit_code=$EXIT" >> $GITHUB_OUTPUT
          if [ "$EXIT" -eq 2 ]; then
            echo "DRIFT DETECTED in ${{ matrix.environment }}"
            cat plan-output.txt
            exit 1
          fi
```

The `-detailed-exitcode` flag makes `terraform plan` return exit code 2 when it finds changes, exit code 0 when there are none, and exit code 1 on error. This makes it trivial to detect drift programmatically. When drift is detected, the CI job fails, which triggers an alert to the on-call channel. The on-call engineer reviews the plan output, determines whether the drift is intentional (an emergency change that now needs to be codified in `.tf` files) or unintentional (a misconfiguration that needs to be reverted), and takes action.

## 9. State operations: the dangerous ones you will eventually need

Beyond the daily `plan` and `apply` cycle, there are a handful of state manipulation commands that every Terraform operator needs to know — and needs to treat with extreme caution. These commands bypass the normal reconciliation loop and directly modify the state file. Used incorrectly, they produce state that does not match reality, which is exactly the condition that causes the most dangerous Terraform failures.

### `terraform state mv` — renaming resources in state

When you refactor your Terraform code — renaming a resource, moving it into a module, extracting it from a module — the old resource address and the new address must be reconciled with state. Without intervention, Terraform sees the old address as "deleted" and the new address as "to be created," and plans a destroy-then-create. For a stateless compute resource, that might be acceptable. For a production database, it is not.

`terraform state mv` renames a resource address in the state file without touching the real infrastructure:

```bash
# Rename a resource that was refactored in code
# Old code: resource "aws_security_group" "eks"
# New code: resource "aws_security_group" "eks_nodes"
terraform state mv \
  aws_security_group.eks \
  aws_security_group.eks_nodes

# Move a resource into a module
# Old code: resource "aws_subnet" "private"
# New code: module.vpc.aws_subnet.private
terraform state mv \
  aws_subnet.private \
  module.vpc.aws_subnet.private

# Move from one module call to another
terraform state mv \
  module.vpc_old.aws_vpc.main \
  module.vpc_new.aws_vpc.main
```

After running `state mv`, the next `terraform plan` should show zero changes for the moved resource. If it shows a replacement, the address did not match correctly — check the module path and resource name carefully.

Two safety rules for `state mv`: always run `terraform plan` after every `state mv` to confirm zero diff on the moved resource, and always take a state backup before any manipulation (`terraform state pull > backup-$(date +%Y%m%d%H%M%S).tfstate`). S3 versioning provides an automatic backup, but an explicit local copy before a complex refactor gives you a fast recovery path if something goes wrong.

### `terraform state rm` — removing resources from state

`terraform state rm` removes a resource from the state file without destroying the real infrastructure. The resource continues to exist in AWS; Terraform simply stops managing it. This is useful when you are handing off management of a resource to a different Terraform root module, or when you want Terraform to "forget" about a resource that was created outside IaC and will continue to be managed outside IaC.

```bash
# Remove a resource from state — it still exists in AWS, just not managed by this Terraform
terraform state rm aws_route53_record.legacy_cname

# Remove an entire module from state
terraform state rm module.deprecated_vpc
```

Use `state rm` carefully. After removing a resource, if the resource address still exists in your `.tf` files, the next `plan` will show it as "to be created." Either remove the resource block from code, or handle the re-import if you want management to continue. The worst outcome is `state rm` followed by an accidental `apply` that creates a duplicate resource alongside the original — you now have two copies of something that should be a singleton, and cleaning up is painful.

### `terraform state pull` and `push` — manual state surgery

For advanced scenarios — recovering from a corrupted state, merging two state files, bulk-editing resource attributes that Terraform will not let you change through normal means — you can pull the raw state JSON, edit it, and push it back:

```bash
# Pull current state to a local file
terraform state pull > state-backup.json

# Edit state-backup.json — dangerous, requires knowing the state schema
# Common legitimate edits: fixing a resource's ID after an import went wrong,
# removing a resource attribute that is blocking a plan

# Push the edited state back (bypasses all validation — be certain)
terraform state push state-backup.json
```

This is the nuclear option. The state JSON schema is undocumented and changes between Terraform versions. Incorrect edits produce invalid state that causes all subsequent `plan` and `apply` commands to fail. Use `state pull/push` only as a last resort, only after backing up the original state, and only after confirming the exact schema change you need to make by looking at a working state file for comparison. For every team that has successfully used `state push` to recover from a disaster, there is another team that used it and created a second disaster.

### The `terraform taint` history and its replacement

In older versions of Terraform (pre-0.15.2), you would use `terraform taint` to mark a resource for forced replacement on the next `apply`. The command modified state to set a "tainted" flag, and the subsequent `plan` would show the resource as `-/+`. This was useful when a resource was in an inconsistent state and needed to be recreated.

`terraform taint` was deprecated in Terraform 0.15.2 and removed in later versions. The replacement is the `-replace` flag on `plan` and `apply`:

```bash
# Mark a specific resource for replacement on this apply
terraform apply -replace=aws_instance.web_server

# Plan with replacement to review the impact first
terraform plan -replace=aws_instance.web_server
```

The `-replace` flag is cleaner than `taint` because it does not modify state — the replace decision is made at plan time and is visible in the plan output before apply runs. This makes it auditable in CI and safe to review before committing.

## 10. Security: what state exposes and how to lock it down

The security posture of your Terraform setup is largely determined by two things: who can read state, and who can run `apply`. Both require explicit engineering.

### What lives in state

Run `terraform show` against any non-trivial state file and you will find the raw value of every sensitive resource attribute. For an RDS instance, that includes `password`, `master_username`, and the connection endpoint. For an IAM user with an access key, it includes `secret_access_key` in plaintext. For a TLS certificate private key provisioned with the `tls_private_key` resource, it includes the entire private key.

Terraform 1.x has a `sensitive = true` marker for variables and outputs, which suppresses values in CLI output. But the state file itself always stores the plaintext value regardless of the sensitive marker. The sensitive marker is a display-only protection. If someone can read your state file, they have all your secrets.

The practical mitigation stack:

1. **S3 bucket encryption**: enable SSE-KMS with a customer-managed key. Without this, state is stored as plaintext bytes in S3.
2. **S3 bucket policy**: explicitly deny `s3:GetObject` for any principal not in the approved list. Use bucket policies, not just IAM policies — IAM policies can be bypassed by resource-based policies if they are more permissive.
3. **CloudTrail + S3 object-level logging**: log every read and write to the state bucket. This gives you an audit trail of who read state and when, which is essential for incident response after a credential leak.
4. **Separate plan and apply IAM roles**: the plan role needs only `s3:GetObject` on state and read access to AWS resources. The apply role needs `s3:PutObject` and `s3:GetObject` plus the AWS permissions to provision resources. Operators should assume the plan role for local work. The apply role should be assumed only by CI/CD after a merge gate.
5. **Do not echo state in CI logs**: `terraform show` in CI outputs state to stdout. Redirect output carefully. The plan job posts the plan output as a PR comment, but the plan output does not include sensitive values (they are redacted as `(sensitive value)`). The `terraform state show` command does expose sensitive values — never run it in CI with output captured to a public log.

The broader principle here connects back to the secrets management practices in the IaC foundations post at [/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative](/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative): secrets should flow through a secrets manager (HashiCorp Vault, AWS Secrets Manager, External Secrets Operator), not through Terraform state. When Terraform provisions an RDS instance, it should set a randomly generated password and write it to Secrets Manager, not embed the password as a Terraform output. The state file still contains the password, but the application reads it from Secrets Manager — reducing the blast radius of a state file exposure to Terraform operators, not the entire application stack.

### The CI service identity

The CI/CD pipeline that runs `terraform apply` holds the keys to your production infrastructure. Protecting that identity is critical. The wrong approach is a long-lived AWS access key stored as a CI secret. Those keys rotate infrequently (if ever), are stored in the CI platform's secret store, and are typically the same key used for every environment including production. A compromised CI credential compromises everything.

The right approach, as mentioned in the CI pipeline section above, is OIDC federation. GitHub Actions, GitLab CI, and CircleCI all support OIDC — the CI runner requests a short-lived STS token from AWS, valid for 15 minutes to one hour, scoped to the specific IAM role. The token cannot be used after the job completes. There is no long-lived credential to steal. The IAM trust policy grants the role only to CI jobs from specific branches or repositories:

```hcl
# IAM trust policy for the apply role — only main branch can assume it
resource "aws_iam_role" "terraform_apply" {
  name = "terraform-apply-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            # Only main branch can assume the apply role
            "token.actions.githubusercontent.com:sub" = "repo:acme/infra:ref:refs/heads/main"
          }
        }
      }
    ]
  })
}
```

This trust policy ensures that even if a CI configuration is modified to try to run `apply` from a feature branch, the AWS STS `AssumeRoleWithWebIdentity` call will be rejected by the condition. Only the main branch can get an apply token.

#### Worked example: the before-after of CI security

Before OIDC federation, a typical CI Terraform setup stored a single `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` pair in GitHub secrets. The key had `AdministratorAccess` on the production account (because the engineers did not know what IAM permissions Terraform needed, and `Administrator` "just worked"). The key rotated annually. During a routine dependency audit, the team found that the same access key had been accidentally logged in a GitHub Actions debug log seven months earlier. The key had been leaked to anyone with read access to the repository logs. The blast radius was unbounded — `AdministratorAccess` on the prod account, for seven months.

After OIDC federation, with separate plan and apply roles, least-privilege IAM policies per role, and per-branch trust conditions:

- Zero long-lived credentials to steal — tokens expire after 15 minutes
- Separate plan role with read-only permissions used for all developer and PR work
- Apply role restricted to main branch, with minimum required permissions (not Administrator)
- Attempted assumes from feature branches fail at the STS level
- Any unusual AssumeRoleWithWebIdentity call triggers a CloudTrail alert

The improvement is qualitative but the risk reduction is substantial. The DORA change-failure-rate metric improved from 12% (that team had a high rate of failed deploys partly because the IAM permissions were too broad and caused unexpected side effects from overly permissive automation) to below 4% after the full CI hardening sprint.

## 10. War story: two incidents that should have been prevented

### The RDS replacement that almost was

In early 2023, a team at a mid-sized fintech was migrating their Terraform configurations to a new naming convention. An engineer renamed the `identifier` field of their production RDS instance from `prod-db` to `fintech-prod-rds-us-east-1`, following the new convention. The change looked small in the PR diff — one attribute changed in one file.

Nobody noticed the `-/+` symbol in the plan output. The plan was posted as a comment on the PR, but the comment was 400 lines long. The destroy annotation was on line 287. The PR was approved by an engineer who skimmed the summary line ("Plan: 1 to add, 0 to change, 1 to destroy") and interpreted "1 to destroy" as the old-named resource being replaced by the new-named resource — which is technically what would happen, but "replaced" means destroyed first, including all data.

The apply started in CI. Six seconds later, the monitoring alert fired: the prod database was unreachable. The on-call engineer killed the CI job 12 seconds into the apply. The destroy had already run. The database was gone.

They restored from the most recent automated snapshot — a 40-minute process. Total data loss: the 12 seconds between the last snapshot and the destroy. They got lucky. The lessons: require explicit approval for any plan containing a `-/+` operation on a stateful resource, and add a CI check that blocks auto-apply if the JSON plan includes any `"delete"` action on RDS, EKS, DynamoDB, or ElastiCache resources.

### The state-locking incident

A different team had their Terraform CI set up correctly — S3 backend, DynamoDB locking — but had not thought carefully about what happens when a CI job is killed mid-apply. When a CI runner is killed (timeout, out-of-memory, manual cancel), Terraform does not always get to release the DynamoDB lock before the process exits. The lock item stays in DynamoDB, marked with the CI job ID of the killed run.

The next CI run fails immediately with "Error acquiring the state lock." Engineers see the error and are unsure whether another apply is genuinely running or whether this is a stale lock. They check — no active apply anywhere. But the lock is still there.

The fix is `terraform force-unlock <lock-id>`. The lock ID is printed in the error message. However, this command bypasses the locking mechanism — if two people run `force-unlock` and `apply` simultaneously, you are back to concurrent-apply corruption. The correct procedure is:

```bash
# Confirm the lock is stale (find the lock ID in the error message)
aws dynamodb get-item \
  --table-name terraform-state-lock \
  --key '{"LockID": {"S": "acme-terraform-state-prod/environments/prod/terraform.tfstate"}}' \
  --region us-east-1

# Check if the CI job that holds the lock is still running
# (verify via CI platform — look up the job ID in the lock's Info field)

# If confirmed stale, release it
terraform force-unlock <lock-id>
```

The team added a CI cleanup step that runs `terraform force-unlock` against stale locks older than 30 minutes, after confirming via the CI API that no corresponding job is running. They also added a CI timeout alert — any apply running longer than 15 minutes (far longer than any legitimate apply in their configuration) fires an alert and auto-cancels the job, which triggers the cleanup process.

## 11. How to reach for this — and when not to

### When modules are worth it

Build a module when you need to instantiate the same infrastructure pattern more than twice. A single-use resource block in a root module does not need to be a module. A VPC pattern used across 5 environments and 3 teams is a module. The rule of three applies: copy it once (tolerable), copy it twice (warning sign), copy it three times (extract a module).

Version modules when they are shared across repositories or teams. A module used only within a single monorepo can use relative paths. A module used by multiple teams or published to the Terraform Registry must use versioned sources.

Do not build a module just because a resource is complex. A single `aws_lb` resource with 20 attributes is not necessarily a module — it is a verbose resource. A module earns its existence by abstracting a combination of resources that always appear together and exposing a clean, opinionated interface to callers. Opinionated defaults are the value of a module; if your module just passes every argument through as a variable, it adds complexity without abstraction benefit.

### When workspaces are not worth it

Do not use workspaces if your environments differ in topology — resources present in prod that do not exist in dev. The conditionals multiply and the module becomes unmaintainable. Use directories instead.

Do not use workspaces to manage environments that have different access controls or belong to different AWS accounts. A workspace is just a different state file — it does not change which AWS account the provider authenticates to. For account-per-environment isolation, use directories with different backend configurations and different IAM roles per directory.

Workspaces are well-suited for temporary feature environments — short-lived environments created for a branch, used for integration testing, and destroyed when the branch merges. Creating a feature environment is as simple as `terraform workspace new feature-branch-42 && terraform apply`. Destroying it is `terraform workspace select feature-branch-42 && terraform destroy && terraform workspace delete feature-branch-42`. This pattern works well precisely because feature environments are structurally identical to the main environments and exist for a bounded time.

### When to delay remote state setup

For a solo developer or a two-person team working on infrastructure for the first time, local state is acceptable for the first few weeks. The overhead of bootstrapping an S3 backend, a KMS key, a DynamoDB table, and the right IAM roles is real. The risk of local state — lost file, concurrent apply — is low when only one person is running Terraform. Switch to remote state the moment a second person needs to run Terraform, or the moment you have production infrastructure that you cannot afford to recreate from scratch.

A fast path for teams that are not yet running on AWS: Terraform Cloud's free tier provides remote state, locking, and a basic run UI for up to five users at no cost. This is a reasonable intermediate step — you get remote state without managing the S3 + DynamoDB bootstrap yourself. When the team grows or the organization's policies require self-hosted infrastructure, migrate to an S3 backend using the `terraform state pull` / `terraform state push` pattern to copy state.

### When not to run `terraform apply` in CI automatically

Auto-apply on merge to `main` is a reasonable pattern for dev and staging environments. It keeps those environments current with code changes without requiring manual intervention. For production, auto-apply is a risk that many teams are not willing to accept, and for good reason — a merge that passes code review and CI tests can still contain a Terraform change that destroys a production resource, as the RDS war story demonstrates.

The safer production pattern is manual-approval apply: CI runs `plan` on merge, posts the plan output to a notification channel (Slack, email), and requires an explicit approval from a designated operator before running `apply`. GitHub Actions supports this with the `environment: production` protection rule requiring one or more required reviewers. The reviewer's job is explicitly to look at the Terraform plan output, not just the code diff, and confirm it is safe to proceed.

Teams that have built high confidence in their plan-review gates — automated checks for dangerous operations, required secondary reviewer, mandatory plan-output acknowledgment — can safely automate prod apply as well. The key metric is your change-failure rate. If your prod Terraform changes fail or cause incidents more than 2–3% of the time, the automated apply is premature. DORA's research, documented in the Accelerate book, suggests that high-performing teams achieve a change-failure rate below 15%; elite performers are below 5%. Track yours.

### The discipline beats the tooling

No amount of tooling saves you from skipping the plan review. Atlantis, Spacelift, and Terraform Cloud all make it easier to review plans — they post them to PRs, enforce required approvals, hold plan artifacts — but none of them prevent a human from approving a plan they did not read. Build the culture of plan review alongside the tooling. Make it explicit: approving a PR that includes a Terraform plan is approving that plan. Require the PR approver to write a comment confirming they read the plan output, not just the code diff.

The teams with the strongest Terraform safety records are not the ones with the most sophisticated tooling. They are the ones where every engineer has personally experienced (or heard the story of) a plan that contained an unexpected destroy, and took the lesson seriously. The war stories in this post exist for that purpose. Build the culture of "I always read the plan completely" before you build the automation that runs `apply` without hesitation.

![Terraform state risk taxonomy showing data risks and operational risks branching from the root state risks node](/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-7.png)

## 12. Comparison: remote state backends

| Backend | Locking | Encryption | Versioning | Best for |
|---|---|---|---|---|
| S3 + DynamoDB | DynamoDB conditional writes | S3 SSE + KMS | S3 versioning | AWS-native teams |
| Terraform Cloud / HCP | Built-in | AES-256 | Built-in | Teams wanting a managed UI |
| Google Cloud Storage | GCS object lock | CMEK | GCS versioning | GCP-native teams |
| Azure Blob Storage | Lease-based | Azure SSE | Blob versioning | Azure-native teams |
| Local | None | None | None | Solo dev, tutorials only |

The S3 + DynamoDB backend is the most widely deployed option for AWS teams because it is entirely self-managed, free to run (cost is negligible — a few cents per month for the DynamoDB writes and S3 storage), and integrates naturally with IAM. The Terraform Cloud / HCP Terraform option adds a web UI, a run history, policy-as-code with Sentinel, and a managed API — valuable for larger organizations that want governance features without building them. The cost is \$20/user/month for the Plus tier (as of mid-2025; check current pricing before committing). OpenTofu — the open-source Terraform fork maintained by the Linux Foundation after HashiCorp's BSL license change — is fully compatible with the S3 backend and DynamoDB locking. Teams running OpenTofu use the same backend configuration as shown above, with zero changes. The choice between Terraform and OpenTofu is a licensing and governance decision, not a technical one for the backend configuration.

## 13. Comparison: workspace-per-env vs directory-per-env (detailed)

| Dimension | Workspace-per-env | Directory-per-env |
|---|---|---|
| State isolation | Separate state files per workspace | Separate state files per directory |
| Infrastructure topology | Must be identical (or conditionals) | Can differ freely |
| PR plan check per env | Hard — single workspace, runs sequentially | Easy — separate CI job per directory |
| Cross-env variable override | `terraform.workspace` string comparison | Separate `.tfvars` per directory |
| Access control per env | Backend IAM only — all workspaces use same role | Separate backend and IAM role per directory |
| Risk of env drift | High — conditionals hide divergence | Low — explicit code difference |
| File maintenance overhead | Low — one root module | Higher — N root modules |
| Recommended for | Small, identical environments | Any env with topology differences |

#### Worked example: migrating from workspaces to directories

A team running three workspaces (`dev`, `staging`, `prod`) in a single root module had accumulated 14 `count = terraform.workspace == "prod" ? 1 : 0` conditionals over 18 months. The root module was 1,800 lines. A new engineer spent two days understanding the workspace topology before they could make their first change. Every plan output was 300 lines because it evaluated all the conditionals across all conditional resources.

After migrating to three directories:

- Root module per environment: 200–350 lines each, readable in under 10 minutes
- Plan output per environment: 30–80 lines, matching only the resources in that environment
- CI time for plan: reduced from 4 minutes (single workspace sequential plan) to 90 seconds per environment (parallel CI jobs per directory)
- Time for a new engineer to make their first change: half a day, not two days

The migration itself took one sprint: copy the root module to three directories, remove conditionals, split `.tfvars` files, and update CI. The state remained in S3 — the state file keys changed but the backend configuration was the same. No resources were re-created; the migration was purely a state and code reorganization.

## 14. Module structure: monolith vs composable

![Monolith Terraform module owning all resources compared to small versioned composable modules with independent blast radius](/imgs/blogs/terraform-in-practice-state-modules-and-workspaces-8.png)

The monolith module starts small. You write a module that provisions a VPC, an EKS cluster, and an RDS instance together because they are always deployed together. Makes sense at week one. At month six, a new service needs EKS but not RDS. At month nine, another service needs RDS but its own VPC. At month twelve, the module has grown to include ALBs, S3 buckets, CloudFront distributions, and SQS queues, all bundled together because "we always need these." Now every `plan` against the module evaluates 90 resources, `apply` takes 12 minutes, and changing the EKS version is terrifying because the same module also manages the production database.

The composable alternative is the monolith's antidote. Each module owns one logical component. `modules/vpc-baseline` owns VPC resources. `modules/eks-cluster` owns the EKS cluster and its node groups. `modules/rds-postgres` owns the database. Root modules compose them. An environment that needs EKS without RDS simply does not include the `rds-postgres` module block. An environment that needs RDS at a different scale passes different variable values to the same module.

The blast radius of a change scales with module size. A change to `modules/eks-cluster` only affects EKS resources. A change to the monolith affects every resource the monolith contains. Smaller modules mean smaller blast radius, faster plans, and more confident operators.

## Key takeaways

1. **Never commit `terraform.tfstate` to Git.** Local state is a liability — it disappears on laptop loss and corrupts under concurrent applies. Set up S3 + DynamoDB remote state before any team member other than the first needs to run Terraform.

2. **State contains secrets.** Treat the state S3 bucket like a secrets vault: KMS encryption, strict IAM, versioning enabled, never share bucket access broadly.

3. **Always read the plan before apply.** Every `-/+` symbol is a destroy followed by a create. On a stateful resource (RDS, EKS, ELB), that is a potential production incident. Build CI gates that block auto-apply when the plan contains `"delete"` actions on critical resources.

4. **Version your modules.** Unversioned modules propagate changes to all callers silently. Pin module versions with `?ref=v2.3.0` or the `version =` constraint, and upgrade deliberately.

5. **Directories beat workspaces when environments differ.** Workspaces are great for structurally identical environments. The moment environments diverge in topology, directories with independent root modules are cleaner, safer, and easier to review.

6. **Run `terraform plan` in CI on every PR.** Post the plan output as a PR comment. A plan review is a mandatory step in the merge process, not optional housekeeping.

7. **Add a scheduled drift detection job.** An nightly `terraform plan -detailed-exitcode` against all environments alerts you to drift before it compounds. Catch emergency console changes the morning after, not three months later.

8. **The `import` command is your path out of legacy hand-built infra.** Use `terraform plan -generate-config-out` to generate resource blocks from real infrastructure, clean them up, verify a zero-diff plan, then commit and apply.

9. **Guard state locking explicitly.** Understand what happens to DynamoDB locks when CI jobs are killed. Build a stale-lock detection and cleanup procedure before you need it.

10. **Modules should be the size of one deployed system concern.** Too large and plans are slow, blast radius is huge, and changes are terrifying. Too small and the module overhead costs more than it buys. A 200–500 line module owning 10–30 resources is typically right.

## Further reading

- [HashiCorp Terraform documentation — Backends](https://developer.hashicorp.com/terraform/language/backend) — the canonical reference for backend configuration, including S3 and DynamoDB setup.
- [HashiCorp Terraform documentation — Modules](https://developer.hashicorp.com/terraform/language/modules) — module composition, the registry, and the `source` attribute reference.
- [Terraform: Up and Running, 3rd Edition — Yevgeniy Brikman](https://www.terraformupandrunning.com/) — the most practical book on production Terraform; the module and workspace chapters are authoritative.
- [Atlantis](https://www.runatlantis.io/) — open-source Terraform pull-request automation; posts plan comments, enforces approvals, serializes applies per workspace.
- [The CI/CD mental model — this series](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the commit→build→test→package→deploy→operate spine and DORA metrics frame that every post in this series builds on.
- [Infrastructure as Code: from ClickOps to declarative](/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative) — the previous IaC post that introduced Terraform fundamentals; this post picks up where that one ends.
- [Managing Terraform safely at scale](/blog/software-development/ci-cd/managing-terraform-safely-at-scale) — the next post in the IaC track, covering Atlantis, Terraform Cloud, policy-as-code with OPA/Sentinel, and the organizational patterns that govern Terraform in a large engineering org.
- [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) — the SRE post on building infrastructure that survives the failure modes described here; the reliability principles that pair with this post's operational practices.
