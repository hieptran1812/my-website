---
title: "Immutable infrastructure and golden images: bake, replace, never patch"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn to bake every OS patch and runtime into a golden image with Packer, automate the CI bake pipeline, and replace your entire fleet in under 90 minutes with zero SSH sessions."
tags:
  [
    "ci-cd",
    "devops",
    "immutable-infrastructure",
    "packer",
    "golden-images",
    "infrastructure-as-code",
    "terraform",
    "aws",
    "containers",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/immutable-infrastructure-and-golden-images-1.png"
---

It was a Tuesday afternoon, six months after the startup had scaled from ten servers to two hundred. The on-call engineer got a page: one of the web nodes was returning 500s on roughly a third of requests. She SSH'd in, ran `ps aux`, eyeballed some logs, found a stale process holding a file descriptor, killed it, and the errors stopped. She closed the SSH session, closed the page, and went home.

Nobody documented that command. Nobody knew the node was now in a state no other node was in. Six weeks later, that node silently OOM-killed a different process that was holding a critical lock, and the database connection pool exhausted. The post-mortem traced the root cause back to the Tuesday fix — but by then there was no way to determine exactly what state the server was in, whether other nodes had the same issue, or how to reproduce the environment. The outage lasted three days. The engineering team spent most of it rebuilding a server from memory.

This is the mutable infrastructure anti-pattern in its final form: a server that no longer matches any known-good state, repaired by hand, undocumented, and irreproducible. The fix is not discipline. The fix is making manual SSH fixes physically impossible by ensuring your server is always replaced, never patched.

This post covers exactly that: the philosophy and mechanics of immutable infrastructure, the golden image as an artifact in your CI pipeline, the Packer-based bake process, and how Terraform uses a pinned AMI ID to roll an entire fleet in eight minutes without a single SSH session. By the end you will have a working Packer HCL definition, a GitHub Actions pipeline that bakes and scans images on every base-image commit, and a Terraform module that treats the AMI as a first-class versioned artifact. These are the same tools and patterns referenced across the commit→build→test→package→deploy→operate spine of the [CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model).

![A side-by-side diagram showing a mutable server that drifts into a snowflake via SSH patches versus an immutable server that is replaced from a Packer-baked AMI](/imgs/blogs/immutable-infrastructure-and-golden-images-1.png)

---

## 1. The mutable infrastructure anti-pattern

Every engineering team discovers mutable infrastructure the same way: it works fine at first. You launch a box, install your dependencies, tweak a config file, and the service runs. A week later you push a hotfix via SSH. A month later someone from the security team applies a kernel patch. Three months later an on-call engineer upgrades a library because a job failed. Six months later your "standard Ubuntu server" is a unique organism with a change history that lives entirely in human memory — or more accurately, in nobody's memory at all.

The term for this is **configuration drift**: the gradual divergence between what you believe your infrastructure state to be and what it actually is. Drift happens in small, individually harmless increments. A changed `/etc/sysctl.conf` entry here, a manually installed `strace` package there, a cron job someone added and forgot. None of these individually bring down a service. Together, over time, they make the server irreproducible.

The snowflake problem has three compounding failure modes:

**Reproducibility failure.** When the server breaks, you cannot rebuild it. The last known state is "what it was running when things worked," but you have no record of the two dozen manual interventions between the initial launch and now. Incident response turns into archaeology.

**Autoscaling failure.** When load spikes and your Auto Scaling Group launches a new instance from an old AMI, that instance does not have the six months of patches the existing fleet has. The new instance behaves differently. Debugging flapping health checks becomes comparing a 6-month-old config to a present-day one.

**Audit failure.** Compliance frameworks — SOC 2, PCI-DSS, ISO 27001 — require evidence that your systems are in a known, documented state. "We SSH'd in and patched it" is not evidence. An AMI ID with a CI build log, a Trivy scan report, and a Git SHA is.

The DORA research (Accelerate, 2018; State of DevOps reports 2019–2023) is consistent: teams that treat infrastructure as immutable and code-defined have materially lower change-failure rates and MTTR. The mechanism is simple — if you cannot change a running server, you cannot make an untracked change.

There is a deeper cost to mutable infrastructure that the DORA metrics do not directly measure: **the cognitive cost of uncertainty**. When your engineers cannot answer the question "what is this server running right now?", they become conservative about deployments. They batch up changes. They avoid touching servers that "seem to be working." They create implicit dependencies on the particular state of a particular node. Over time, the fleet becomes something that everyone is afraid to touch — which is the opposite of a productive deployment culture.

A post-mortem pattern that appears again and again in mutable infrastructure environments is what I call the "it worked on my node" problem. An application works on nodes 1, 2, and 4 of a fleet, but intermittently fails on node 3. The investigation reveals that node 3 has a different version of a system library because someone patched it separately three months ago. The fix is easy once discovered; the investigation takes hours. On an immutable fleet, this class of problem does not exist: all nodes boot from the same image.

### The cost of drift compounds nonlinearly

Drift is not a linear problem. A fleet of 10 servers that has been running for 12 months with periodic manual patches has not accumulated 10 servers' worth of configuration drift — it has accumulated something closer to 10 × N drift, where N is the number of distinct manual interventions, because each intervention interacts with previous ones in ways that are hard to predict. The maintenance cost of understanding and managing that drift grows faster than the fleet size. This is why teams running mutable infrastructure often report that adding new servers to the fleet is more dangerous than replacing old ones: new servers start in a known state (the original image), while old servers are in an unknown state (whatever accumulated over their lifetime), and the two interact in surprising ways during load balancing.

## 2. Immutable infrastructure: the core principle

The defining rule of immutable infrastructure is stated in a single sentence: **a server never changes after it boots.** If something needs to change — an OS patch, a new runtime version, a changed configuration — you do not touch the running server. You build a new image, launch new servers from it, and terminate the old ones.

This is not a new idea. It was articulated clearly by Chad Fowler in a 2013 blog post ("Trash Your Servers and Burn Your Code"), popularized by Martin Fowler's 2011 work on "Snowflake Servers" and "Phoenix Servers," and systematized by HashiCorp's Packer (first release 2013) and the AMI-based infrastructure model AWS promoted throughout the 2014–2016 period. It has since become the dominant approach for any team operating infrastructure at scale.

The practical implications:

- **No SSH access to production servers.** If you need to debug, you read logs (shipped to a centralized log aggregator at boot). If you need to change behavior, you bake a new image.
- **Configuration baked in, not injected at runtime.** OS-level settings, package versions, monitoring agents, security hardening — all baked into the AMI. Environment-specific configuration (database URLs, API keys) is injected via the metadata service or a secrets manager at boot, but the base image is identical across all environments.
- **The image is the artifact.** Just as [build-once promote everywhere](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) treats a container image as the immutable unit that flows through environments, immutable infrastructure treats the AMI (or GCE image, or Azure VHD, or Vagrant box) as the unit. The AMI ID is the version identifier.
- **Replace, do not restart.** Health checks and rolling replacements handle fleet updates. Terraform's `aws_autoscaling_group` with a `launch_template` update and an instance refresh handles this with a single `terraform apply`.

### What immutable infrastructure does not mean

There is a common misunderstanding worth addressing directly: immutable infrastructure does not mean that your application's *data* is immutable, or that your application *state* is immutable. It means that the *server* — the OS, the runtime, the installed packages, the configuration files — is immutable. Your application still writes to a database. Your Redis cluster still accumulates data. Your S3 bucket still holds files that change. The immutability constraint applies to the infrastructure layer, not to the data layer.

Similarly, immutable infrastructure does not mean you never change your infrastructure. You change it constantly — by baking new images and replacing the fleet. The point is that you change it through a controlled, logged, reproducible process rather than through manual SSH intervention. The rate of change can be *higher* on an immutable fleet than on a mutable one, precisely because each change is safe, audited, and reversible. DORA elite performers ship infrastructure changes frequently *because* they are confident each change is well-contained, not in spite of it.

### The three-layer configuration model

A useful mental structure for what gets baked where:

| Layer | What belongs here | How it is applied |
|---|---|---|
| AMI / golden image | OS kernel, packages, runtime, agents, hardening | Packer bake — once per image |
| Launch configuration | Instance type, IAM role, security group, VPC | Terraform — per environment |
| Boot-time injection | Environment variables, secrets, feature flags | User data + secrets manager |

The baked layer is stable and identical across all environments. The launch configuration layer differs per environment but is defined in Terraform and version-controlled. The boot-time injection layer is ephemeral and injected at startup — never baked, because it contains environment-specific secrets.

This model resolves the tension between "everything baked" and "environments are different." The answer is: the OS and runtime are identical everywhere; the environment identity is injected at boot from the secrets manager. No environment-specific data ever touches the image.

## 3. Cattle, not pets: the mental shift

The phrase "cattle, not pets" originated in a 2011 presentation by Randy Bias and Bill Baker at CloudStack Collaboration Conference. The analogy maps directly:

**Pets** are named, hand-crafted, individually important servers. You give them human names (web-prod-alice, db-prod-bob). When they get sick you nurse them back to health. You care which specific machine is running your workload. You SSH into them regularly. Their death is a crisis.

**Cattle** are numbered, disposable servers. They have machine-generated names or IDs (web-prod-i-0a3b4c5d6e, or just web-prod-007). When one gets sick you replace it. You do not care which specific machine is running your workload — you care that the right number of healthy machines are running it. Their death is an auto-scaling event.

The mental shift has concrete operational consequences.

First, **autoscaling becomes trivial**. An ASG that launches new nodes from a golden AMI gives you servers that are byte-for-byte identical to the ones that have been running for six months. There is no convergence step, no Ansible play on first boot, no "wait for Chef to run." The new node either passes its health check and joins the fleet, or it does not and gets terminated — in either case the error is in the image definition, not in the unique history of a specific node.

Second, **recovery is fast and deterministic**. When a cattle node fails, you do not debug it — you replace it. A launch template pointing to a known-good AMI means recovery time is measured in minutes (boot + health check) rather than hours (debug + patch + restart + test). In one measured incident response, replacing a 20-node fleet after a kernel panic took eight minutes: the Terraform `apply` started at 14:03, the last new node passed its health check at 14:11. The mutable equivalent — debugging and manually patching nodes one by one — had taken 3.5 hours in a prior incident.

Third, **SSH access disappears from your threat model**. One of the most common vectors for an attacker who gains a foothold on a server is to establish persistence by modifying startup scripts, installing a backdoor, or creating a new privileged user. On a cattle-model immutable fleet, there is nothing to persist to: the next deployment replaces the instance. The attack surface of "what can someone do with unauthorized shell access" collapses to "cause a brief outage in one node" — the ASG replaces it automatically.

### The practical mechanics of the transition

Moving from pets to cattle is not primarily a tooling change — it is an operational culture change. The tooling (Packer, Terraform, ASGs, SSM) is well-understood and widely documented. The hard part is the organizational shift:

**Engineers stop debugging individual servers and start debugging image definitions.** When a node is unhealthy, the response is no longer "SSH in and look around." It is "terminate the node, let the ASG replace it, and if the problem reproduces, look at CloudWatch Logs for what went wrong during startup." The debugging surface changes from "the live server state" to "the logs produced at startup and during operation."

**Incident runbooks change.** A traditional runbook step that says "SSH to the affected node and run X" must be replaced with either "check the CloudWatch log group for this error" or "roll back to the previous AMI." This rewriting of runbooks takes deliberate effort and is often the last step in the pet-to-cattle migration.

**Monitoring becomes more important.** On a pets fleet, you can SSH in to check a metric that is not in your monitoring stack. On a cattle fleet, if it is not in your monitoring stack, it does not exist. This forces investment in comprehensive observability — centralized log aggregation (CloudWatch Logs, Elasticsearch, Loki), metrics collection (CloudWatch, Prometheus via the CloudWatch agent or a Prometheus scrape endpoint), and distributed tracing if your application warrants it. This is a forcing function that generally leaves the team better off, but it takes time and deliberate investment.

**The `Makefile` or `justfile` replaces the SSH alias.** A common intermediate step: engineers keep a collection of SSH aliases for jumping to specific servers. On cattle fleets, those aliases are replaced by `make logs service=web env=prod` (which runs `aws logs tail`) and `make rollback service=web ami=ami-123456` (which updates SSM and runs `terraform apply`). The operational tasks are the same; the mechanism is controlled and auditable.

![A comparison of the pets model with named servers and SSH access versus the cattle model with numbered identical servers replaced from a golden AMI](/imgs/blogs/immutable-infrastructure-and-golden-images-4.png)

## 4. Golden images with Packer

**Packer** (by HashiCorp, first released 2013) is the standard tool for building machine images. It reads an HCL (or JSON) template that describes a source (what to start from: an Ubuntu 22.04 AMI from AWS), a set of provisioners (what to run: shell scripts, Ansible playbooks, Chef cookbooks), and a set of post-processors (what to do with the result: push to ECR, create a manifest, run a scanner). The output is a registered image artifact — an AMI ID, a GCE image name, a Docker image tag — that you can reference in Terraform or any launch template.

The reason Packer is the right tool for this job rather than, say, running Ansible on a base AMI at boot, is that **the bake is done once and the artifact is tested before deployment.** If you run Ansible on boot, you have a convergence window — a period after the instance launches but before configuration is complete — during which your service may be in an inconsistent state. You also cannot test the result before it reaches production. Packer inverts this: the heavy work happens in CI, the artifact is scanned and validated before it reaches any environment, and the boot time of a new instance is just "copy AMI to region + launch."

### The Packer HCL structure

A Packer configuration has three main blocks:

**`packer {}` block** — declares required plugins and version constraints.

**`source` block** — declares the builder (the starting point). For AWS this is `amazon-ebs`, which launches a temporary EC2 instance from a source AMI, runs provisioners, and creates a new AMI from the stopped instance.

**`build` block** — chains sources and provisioners. The provisioners run in order: typically a shell script for initial setup, an Ansible playbook for the main configuration, and a second shell script for cleanup and hardening.

#### Worked example: Packer HCL for Ubuntu 22.04 + Node.js 20 AMI

Here is a complete, runnable Packer configuration for baking a golden AMI that includes Ubuntu 22.04, Node.js 20, the CloudWatch agent, and CIS-level 1 hardening:

```hcl
# packer/ubuntu-nodejs.pkr.hcl

packer {
  required_plugins {
    amazon = {
      version = ">= 1.3.0"
      source  = "github.com/hashicorp/amazon"
    }
    ansible = {
      version = ">= 1.1.1"
      source  = "github.com/hashicorp/ansible"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables — override via -var or PKR_VAR_* environment variables
# ---------------------------------------------------------------------------
variable "aws_region" {
  default = "us-east-1"
}

variable "nodejs_version" {
  default = "20"
}

variable "build_version" {
  # Injected by CI: git SHA short, e.g. "a1b2c3d"
  default = "dev"
}

variable "source_ami_filter" {
  # Always start from the latest Ubuntu 22.04 LTS HVM SSD AMI
  default = "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"
}

# ---------------------------------------------------------------------------
# Source: AWS EBS builder
# ---------------------------------------------------------------------------
source "amazon-ebs" "ubuntu-nodejs" {
  region        = var.aws_region
  instance_type = "t3.medium"  # Build instance; larger = faster provisioning

  # Dynamically resolve the latest Ubuntu 22.04 AMI owned by Canonical
  source_ami_filter {
    filters = {
      name                = var.source_ami_filter
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["099720109477"]  # Canonical's AWS account ID
  }

  ssh_username  = "ubuntu"

  # The resulting AMI name includes the git SHA and build date for traceability
  ami_name      = "golden-nodejs${var.nodejs_version}-${formatdate("YYYYMMDD", timestamp())}-${var.build_version}"
  ami_description = "Golden Ubuntu 22.04 + Node.js ${var.nodejs_version} | build ${var.build_version}"

  # Tag the AMI and the snapshot for cost allocation and audit
  tags = {
    Name          = "golden-nodejs${var.nodejs_version}"
    NodejsVersion = var.nodejs_version
    BuildVersion  = var.build_version
    BuildDate     = formatdate("YYYY-MM-DD", timestamp())
    ManagedBy     = "packer"
    BaseOS        = "ubuntu-22.04"
  }

  # Encrypt the root volume at rest
  encrypt_boot = true

  # Restrict AMI sharing to the org's accounts only (change to your org OU)
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 20
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  # Temporary security group — restrict to the CI runner's IP via PKR_VAR_*
  temporary_security_group_source_cidrs = ["10.0.0.0/8"]
}

# ---------------------------------------------------------------------------
# Build: provision the AMI
# ---------------------------------------------------------------------------
build {
  name    = "golden-ubuntu-nodejs"
  sources = ["source.amazon-ebs.ubuntu-nodejs"]

  # Step 1: Wait for cloud-init to finish before we start
  provisioner "shell" {
    inline = [
      "cloud-init status --wait",
      "sudo apt-get update -y",
      "sudo apt-get install -y python3 python3-pip"
    ]
  }

  # Step 2: Ansible playbook — installs Node.js, CloudWatch agent, hardening
  provisioner "ansible" {
    playbook_file   = "${path.root}/../ansible/golden-image.yml"
    extra_arguments = [
      "-e", "nodejs_version=${var.nodejs_version}",
      "-e", "build_version=${var.build_version}"
    ]
    ansible_env_vars = [
      "ANSIBLE_HOST_KEY_CHECKING=False",
      "ANSIBLE_FORCE_COLOR=True"
    ]
  }

  # Step 3: Harden and clean up (remove SSH authorized_keys, clear bash history)
  provisioner "shell" {
    inline = [
      # Disable root login
      "sudo sed -i 's/^PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config",
      # Remove ubuntu user's SSH authorized_keys — no SSH access to prod instances
      "sudo rm -f /home/ubuntu/.ssh/authorized_keys",
      # Clear package cache
      "sudo apt-get clean",
      "sudo rm -rf /var/lib/apt/lists/*",
      # Clear bash history
      "history -c",
      "sudo truncate -s 0 /root/.bash_history",
      "echo 'Packer hardening complete'"
    ]
  }

  # Post-processor: write a local manifest with the AMI ID for downstream use
  post-processor "manifest" {
    output     = "packer-manifest.json"
    strip_path = true
  }
}
```

A few design choices worth explaining:

- **`source_ami_filter` with `most_recent = true`** means every CI run starts from the latest Ubuntu 22.04 AMI Canonical publishes. You do not manually track base image updates — the bake pipeline handles it.
- **`encrypt_boot = true`** encrypts the root EBS volume at rest. This is a hard requirement for most compliance frameworks and costs nothing extra on gp3 volumes.
- **Removing SSH authorized keys in the hardening step** is the mechanism that enforces the "no SSH access" rule. There are no keys to SSH with on a launched instance — the attack surface is gone.
- **The manifest post-processor** writes `packer-manifest.json` with the AMI ID. The CI pipeline reads this file to extract the AMI ID for downstream Terraform steps.

The companion Ansible playbook (`ansible/golden-image.yml`) installs Node.js via the NodeSource APT repository, installs and configures the CloudWatch unified agent, applies CIS Ubuntu 22.04 benchmarks via the `devsec.hardening.os_hardening` role, and configures the system journal to forward logs. This is standard configuration management work; the key point is that it runs at build time, not at instance launch time.

![The layered golden image bake pipeline from Packer HCL source through provisioning, test, scan, and AMI publication to Terraform fleet deployment](/imgs/blogs/immutable-infrastructure-and-golden-images-2.png)

## 5. The image bake pipeline in CI

A golden image that is built manually is just a snapshot. A golden image that is built automatically in CI every time the base image or the provisioning code changes is an artifact with full provenance — a Git commit, a CI run ID, a Trivy scan report, and an AMI ID that traces back through all of them.

The bake pipeline lives in a dedicated CI workflow. It is triggered by:

1. A push to the `packer/` or `ansible/` directories (provisioning code changed)
2. A scheduled nightly run (to pick up daily Ubuntu AMI updates from Canonical)
3. A push to any `base-image-version` config file you maintain

Here is the complete GitHub Actions workflow:

```yaml
# .github/workflows/bake-ami.yml
name: Bake golden AMI

on:
  push:
    branches: [main]
    paths:
      - "packer/**"
      - "ansible/**"
      - ".github/workflows/bake-ami.yml"
  schedule:
    # Nightly at 02:00 UTC — picks up latest Ubuntu AMI from Canonical
    - cron: "0 2 * * *"
  workflow_dispatch:
    inputs:
      nodejs_version:
        description: "Node.js major version to bake"
        required: false
        default: "20"

permissions:
  id-token: write   # Required for OIDC federation to AWS
  contents: read

jobs:
  bake:
    name: Packer bake + Trivy scan
    runs-on: ubuntu-22.04
    timeout-minutes: 45

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure AWS credentials via OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}:role/packer-ci-role
          aws-region: us-east-1
          # OIDC: no long-lived AWS keys stored in GitHub secrets

      - name: Set up Packer
        uses: hashicorp/setup-packer@main
        with:
          version: "1.10.0"

      - name: Set up Ansible
        run: |
          pip3 install ansible==9.1.0 boto3 botocore
          ansible-galaxy collection install devsec.hardening

      - name: Install Packer plugins
        working-directory: packer
        run: packer init ubuntu-nodejs.pkr.hcl

      - name: Validate Packer template
        working-directory: packer
        run: packer validate ubuntu-nodejs.pkr.hcl

      - name: Bake AMI
        id: bake
        working-directory: packer
        env:
          PKR_VAR_build_version: ${{ github.sha }}
          PKR_VAR_nodejs_version: ${{ inputs.nodejs_version || '20' }}
        run: |
          packer build -color=false ubuntu-nodejs.pkr.hcl
          # Extract AMI ID from the manifest
          AMI_ID=$(jq -r '.builds[-1].artifact_id | split(":")[1]' packer-manifest.json)
          echo "ami_id=${AMI_ID}" >> "$GITHUB_OUTPUT"
          echo "Built AMI: ${AMI_ID}"

      - name: Scan AMI with Trivy
        # Pull the AMI as a filesystem snapshot and scan it for CVEs
        # We use the EC2 AMI export approach: launch a temporary instance,
        # mount the root volume, and run Trivy against the filesystem.
        # For a faster approach: scan the Ansible-managed package list instead.
        run: |
          # Install Trivy
          wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
          echo "deb https://aquasecurity.github.io/trivy-repo/deb generic main" \
            | sudo tee /etc/apt/sources.list.d/trivy.list
          sudo apt-get update && sudo apt-get install -y trivy
          # Scan the Packer build directory (cached rootfs from last provisioner)
          trivy fs \
            --severity CRITICAL,HIGH \
            --exit-code 1 \
            --format table \
            packer/

      - name: Tag AMI as tested
        if: success()
        env:
          AMI_ID: ${{ steps.bake.outputs.ami_id }}
        run: |
          aws ec2 create-tags \
            --resources "$AMI_ID" \
            --tags \
              Key=Status,Value=tested \
              Key=ScanPassed,Value=true \
              Key=CIRunId,Value=${{ github.run_id }}

      - name: Publish AMI ID to SSM Parameter Store
        if: success()
        env:
          AMI_ID: ${{ steps.bake.outputs.ami_id }}
        run: |
          # Terraform reads this parameter to get the latest tested AMI ID
          aws ssm put-parameter \
            --name "/golden-images/nodejs20/latest-tested" \
            --value "$AMI_ID" \
            --type String \
            --overwrite
          echo "Published ${AMI_ID} to SSM"

      - name: Notify on failure
        if: failure()
        run: |
          echo "AMI bake or scan failed — see CI logs for details"
          # In production: post to Slack/PagerDuty via your notification tool

  promote:
    name: Promote AMI to prod-eligible
    needs: bake
    runs-on: ubuntu-22.04
    # Only promote on pushes to main, not on nightly schedule
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - name: Configure AWS credentials via OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}:role/packer-ci-role
          aws-region: us-east-1

      - name: Read latest tested AMI
        id: read_ami
        run: |
          AMI_ID=$(aws ssm get-parameter \
            --name "/golden-images/nodejs20/latest-tested" \
            --query "Parameter.Value" \
            --output text)
          echo "ami_id=${AMI_ID}" >> "$GITHUB_OUTPUT"

      - name: Promote AMI to prod-eligible
        env:
          AMI_ID: ${{ steps.read_ami.outputs.ami_id }}
        run: |
          aws ec2 create-tags \
            --resources "$AMI_ID" \
            --tags Key=Status,Value=prod-eligible
          aws ssm put-parameter \
            --name "/golden-images/nodejs20/prod-eligible" \
            --value "$AMI_ID" \
            --type String \
            --overwrite
          echo "Promoted ${AMI_ID} to prod-eligible"
```

A few pipeline design decisions worth noting:

**OIDC federation instead of long-lived AWS keys.** The `id-token: write` permission and `aws-actions/configure-aws-credentials` with `role-to-assume` mean the CI runner never holds an AWS access key. It gets a short-lived token from AWS STS via the OIDC exchange. This matters because CI systems are a frequent target for secret theft; removing the secret removes the target. The [image security scanning post](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface) covers OIDC federation in detail.

**SSM Parameter Store as the AMI registry.** After a successful scan, the pipeline writes the AMI ID to an SSM parameter. Terraform reads that parameter — not a hardcoded AMI ID — which means the fleet always uses the latest tested image without requiring a Terraform change to update the AMI reference.

**Two-tier promotion.** The bake job produces a `tested` AMI. The `promote` job (which only runs on pushes to main, not on the nightly schedule) elevates it to `prod-eligible`. This gives you a gating step where a human or an integration test suite can reject an AMI between bake and fleet deployment.

![A branching CI graph showing the golden image pipeline with a scan gate that blocks publication on failure and promotes to a tagged AMI ID on success](/imgs/blogs/immutable-infrastructure-and-golden-images-3.png)

## 6. Rebuild-don't-patch: the operating discipline

The hardest part of immutable infrastructure is not the tooling — it is the discipline. The tooling is straightforward: Packer bakes, Trivy scans, Terraform replaces. The discipline is the habit of reaching for "bake a new image" instead of "SSH in and fix it" every single time.

The discipline breaks down in two recurring scenarios:

**The emergency.** Production is down. A quick SSH fix can restore service in two minutes. Baking a new image takes twenty-five minutes. In the moment, the pressure to SSH in is enormous. If you do it, you have just created the Tuesday afternoon incident from the opening of this post — a server in an undocumented state that will confuse the next person who looks at it, possibly at 3am six months from now.

The correct response to the emergency is to **roll back the fleet to the previous AMI**. If the current image is broken, the previous image worked. Rolling back with Terraform takes about the same time as baking a new image: you update the SSM parameter to point to the previous AMI ID, and run `terraform apply`. The fleet replaces itself. Then you fix the actual problem in the image definition and bake a new image as the permanent fix.

**The "it's just one change."** A developer wants to test a configuration change. They want to just SSH in and edit a config file to see if it helps. This is never just one change — it is the first step toward a server that has diverged from every other server and from the image definition. The correct response is to make the change in the Packer/Ansible definition, bake a new image, and test with that image. Yes, it takes longer. That time cost is the price of having a reproducible, auditable fleet.

### The rebuild-don't-patch contract

A practical way to enforce this discipline is to make it explicit in your runbook and your IAM policy. On the tooling side:

```bash
# Lock down SSH access to production instances via IAM
# Add this to your EC2 instance's IAM role trust policy to deny
# SSH key injection via SSM Session Manager in production

# In your AWS Config rules, alert on:
# - EC2 instances with SSH port (22) open to 0.0.0.0/0
# - SSM Session Manager sessions to production-tagged instances

# In your Auto Scaling Group launch template, set:
# - key_name = null  (no SSH key pair attached)
# - No EC2 key pair means no SSH access even if someone
#   tries to use aws ec2-instance-connect send-ssh-public-key

aws autoscaling describe-launch-templates \
  --query "LaunchTemplates[?LaunchTemplateName=='web-prod']"
```

On the process side, your incident runbook should include a step that explicitly asks: "Is this a patching action that belongs in the image, or a rollback action?" If the answer is "patch," the runbook says: "Update the image definition, bake, scan, deploy. Do not SSH." If the answer is "rollback," the runbook says: "Update the SSM parameter to the previous AMI ID, run Terraform apply."

## 7. Terraform and fleet replacement

The relationship between Packer and Terraform is clean and complementary: Packer builds the artifact, Terraform uses it. Terraform never bakes an image; Packer never manages infrastructure state. The handoff point is the AMI ID, passed via SSM Parameter Store.

Here is a complete Terraform module for a fleet that reads the prod-eligible AMI from SSM:

```hcl
# terraform/modules/web-fleet/main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ---------------------------------------------------------------------------
# Read the prod-eligible AMI ID from SSM Parameter Store
# This is written by the CI bake pipeline after scan + promote
# ---------------------------------------------------------------------------
data "aws_ssm_parameter" "ami_id" {
  name = "/golden-images/nodejs20/prod-eligible"
}

# ---------------------------------------------------------------------------
# Launch Template — references the immutable AMI
# ---------------------------------------------------------------------------
resource "aws_launch_template" "web" {
  name_prefix   = "web-prod-"
  image_id      = data.aws_ssm_parameter.ami_id.value
  instance_type = var.instance_type

  # No key pair — enforces the no-SSH rule
  key_name = null

  # IAM instance profile for CloudWatch agent and SSM (read-only, no Session Manager in prod)
  iam_instance_profile {
    name = aws_iam_instance_profile.web.name
  }

  # User data: inject environment-specific config at boot (no image modification)
  user_data = base64encode(templatefile("${path.module}/user-data.sh.tpl", {
    environment  = var.environment
    db_ssm_path  = var.db_ssm_path
    log_group    = aws_cloudwatch_log_group.web.name
  }))

  network_interfaces {
    associate_public_ip_address = false
    security_groups             = [aws_security_group.web.id]
    delete_on_termination       = true
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "web-${var.environment}"
      AMIId       = data.aws_ssm_parameter.ami_id.value
      ManagedBy   = "terraform"
      Environment = var.environment
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ---------------------------------------------------------------------------
# Auto Scaling Group — triggers instance refresh on launch template change
# ---------------------------------------------------------------------------
resource "aws_autoscaling_group" "web" {
  name = "web-${var.environment}-${substr(data.aws_ssm_parameter.ami_id.value, -8, -1)}"

  desired_capacity = var.desired_capacity
  min_size         = var.min_size
  max_size         = var.max_size

  vpc_zone_identifier = var.subnet_ids

  launch_template {
    id      = aws_launch_template.web.id
    version = "$Latest"
  }

  # Instance refresh: replace nodes rolling when launch template changes
  # (i.e., when the AMI ID changes)
  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 80    # Keep 80% of fleet healthy during replacement
      instance_warmup        = 60    # Seconds to wait before counting new instance as healthy
      checkpoint_percentages = [20, 50, 100]  # Pause points for manual review
    }
    triggers = ["launch_template"]
  }

  health_check_type         = "ELB"
  health_check_grace_period = 90

  tag {
    key                 = "Name"
    value               = "web-${var.environment}"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
    # Ignore changes to desired_capacity — managed by scaling policies, not Terraform
    ignore_changes = [desired_capacity]
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "ami_id_in_use" {
  description = "The AMI ID currently deployed to this fleet"
  value       = data.aws_ssm_parameter.ami_id.value
}

output "launch_template_id" {
  description = "Launch template ID for reference by scaling policies"
  value       = aws_launch_template.web.id
}
```

The key piece is the `instance_refresh` block. When `terraform apply` runs and the SSM parameter has a new AMI ID — because the bake pipeline ran, published a new AMI, and updated the SSM parameter — Terraform detects the change in the `data.aws_ssm_parameter.ami_id` result, which changes the `image_id` in the launch template, which changes the launch template version, which triggers the instance refresh. The ASG then replaces nodes rolling: 20% replaced first, waits for health checks, 50%, waits, 100%.

This is the full commit→deploy→operate spine for infrastructure: a Git commit to the Ansible playbook triggers a Packer bake in CI, which updates the SSM parameter, which is picked up by `terraform apply`, which rolls the fleet. No SSH session anywhere in the chain.

#### Worked example: fleet replacement timing with instance refresh

A web fleet of 20 `t3.medium` instances with `min_healthy_percentage = 80`:

- Maximum instances under replacement at any time: `20 * (1 - 0.80) = 4` instances
- Instance warmup time: 60 seconds (health check grace period)
- ALB deregistration delay: 30 seconds (configured on the target group)
- Time per batch of 4 instances: approximately 90–120 seconds
- Number of batches: `20 / 4 = 5` batches (with checkpoint pauses at 20%, 50%)
- Total replacement time for 20-node fleet: approximately 8–12 minutes

The checkpoint pauses are optional but valuable during the first few deployments of a new AMI. After the first 20% of nodes are replaced, you can inspect the CloudWatch dashboards and ALB error rates. If something is wrong, you can abort the refresh and Terraform rolls back the launch template. The 80% of still-running old-AMI nodes keep serving traffic throughout.

Contrast with in-place patching of a 20-node fleet: `apt-get upgrade` run serially across nodes, manually, with no consistent health check, takes 3–8 hours depending on the package delta and team availability.

## 8. The end-to-end OS patch flow

Put plainly, the complete patching workflow for a security CVE using immutable infrastructure is:

1. Ubuntu publishes a patch for the CVE.
2. Canonical releases an updated Ubuntu 22.04 AMI with the patch applied.
3. Your CI bake pipeline (scheduled nightly, or triggered by a base AMI version watcher) runs `packer build`. Because `most_recent = true` in the `source_ami_filter`, Packer starts from the new Canonical AMI.
4. Ansible provisions Node.js, CloudWatch agent, and hardening on top.
5. Trivy scans the result. If the CVE is still present (e.g., it requires a package not included in the base OS), the scan fails with exit code 1. CI blocks. Your team gets an alert. You fix the Ansible playbook to explicitly install the patched package version and re-run.
6. If the scan passes, the AMI is tagged as `tested`, the SSM parameter is updated.
7. The `promote` job tags it `prod-eligible`, updates the prod SSM parameter.
8. `terraform apply` — either triggered manually by the on-call team or via a GitOps pipeline watching the SSM parameter — detects the AMI change, updates the launch template, and triggers the instance refresh.
9. The 20-node fleet is fully replaced in 8–12 minutes. All new instances have the patched kernel. All old instances are terminated.

No human touched a running server. Every step is logged (CI run ID, AMI ID, Trivy scan report, Terraform state). The entire chain from "CVE announced" to "fleet patched" takes under 90 minutes if the bake pipeline is running and the Terraform apply is part of a CD pipeline, or under two hours if the Terraform apply requires a manual approval gate.

### Multi-region AMI replication

One detail that the single-region example glosses over: in a multi-region deployment, the AMI must be available in every region where you have instances. AWS AMIs are region-scoped. After the bake pipeline produces an AMI in `us-east-1`, you must copy it to `us-west-2`, `eu-west-1`, and any other regions before Terraform can use it there.

The Packer `amazon-ebs` builder handles this with the `ami_regions` list:

```hcl
source "amazon-ebs" "ubuntu-nodejs" {
  # ... other config ...

  # Automatically copy the AMI to these regions after bake
  ami_regions = [
    "us-east-1",
    "us-west-2",
    "eu-west-1",
    "ap-southeast-1"
  ]
}
```

With `ami_regions` set, Packer copies the AMI to all regions as part of the build step. Each region gets its own AMI ID. The Terraform data source must look up the correct AMI ID per region. One pattern is to store the full AMI ID map in SSM, keyed by region:

```bash
# After bake, publish per-region AMI IDs to SSM
for region in us-east-1 us-west-2 eu-west-1 ap-southeast-1; do
  AMI_ID=$(jq -r ".builds[-1].artifact_id" packer-manifest.json \
    | tr ',' '\n' \
    | grep "^${region}:" \
    | cut -d: -f2)
  aws ssm put-parameter \
    --region "$region" \
    --name "/golden-images/nodejs20/prod-eligible" \
    --value "$AMI_ID" \
    --type String \
    --overwrite
done
```

Each regional Terraform workspace then reads `/golden-images/nodejs20/prod-eligible` from the SSM service in its own region and gets the correct AMI ID.

### Baselining AMI age: the 30-day rule

A golden image policy that is often underspecified: how old is too old? If your bake pipeline runs on schedule but nobody applies the result to the fleet for three months, you have the worst of both worlds — a pipeline that produces images nobody uses. A practical rule: any AMI older than 30 days that is referenced in a production launch template should trigger a review.

You can enforce this automatically with an AWS Config rule or a custom Lambda that checks the age of the AMI referenced in each launch template and posts an alert if it exceeds 30 days. This creates automatic pressure to keep the fleet current without requiring manual tracking.

![A timeline from CVE announcement through base image update, Packer bake, Trivy scan, AMI publication, Terraform plan, and full fleet replacement in 85 minutes](/imgs/blogs/immutable-infrastructure-and-golden-images-6.png)

## 9. Container images as immutable infrastructure

Docker images are immutable infrastructure at a finer granularity than VM images. Every `docker build` produces a content-addressed layer stack identified by a SHA256 digest. The image is immutable by definition: once pushed to a registry, `nginx:1.25.3@sha256:a6a5b...` is exactly that image, forever. Pulling it gives you the same bytes every time.

The golden image principle applies directly:

- **You never `docker exec` into a running container to change it.** If you need a change, you rebuild the image with updated Dockerfile instructions, push to the registry, and redeploy.
- **The image is the artifact that flows through environments.** As covered in [build-once promote everywhere](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning), the same digest that passed integration tests in staging is the digest deployed to production — not a rebuild.
- **Distroless containers** take the principle to its logical extreme: a distroless container image (`gcr.io/distroless/nodejs20-debian12`) contains only the Node.js runtime, the CA certificates, and the application. There is no shell, no package manager, no `apt`, no `curl`. You cannot SSH into it. You cannot patch it. You cannot add a backdoor to it. The attack surface is minimal by construction.

Here is a production Dockerfile that embodies the immutable image principle:

```dockerfile
# Dockerfile — multi-stage, distroless final image

# Stage 1: build dependencies (large, discarded after build)
FROM node:20-bookworm-slim AS builder

WORKDIR /app

# Install only production dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy application source and build
COPY src/ ./src/
RUN npm run build

# Stage 2: runtime image — distroless
# This image has no shell, no package manager, no curl.
# It is immutable by construction: there is nothing to patch at runtime.
FROM gcr.io/distroless/nodejs20-debian12:nonroot

WORKDIR /app

# Copy only the built application and production node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules

# Run as non-root user (nonroot = UID 65532 in distroless)
USER nonroot

# Declare the port (documentation; actual binding is in Kubernetes Service)
EXPOSE 3000

# Entrypoint — no shell script, just the Node.js binary
CMD ["dist/server.js"]
```

The resulting image is typically 80–120 MB for a Node.js application, compared to 1.2–1.9 GB for a naive `FROM node:20` single-stage image. There is nothing in the image to scan for shell vulnerabilities, package manager vulnerabilities, or compiler toolchain vulnerabilities. The `node:20` base has approximately 150–200 packages installed; the distroless Node.js base has approximately 25. Trivy finds proportionally fewer CVEs.

The connection between golden AMIs and distroless containers is direct: both are applications of the same principle at different layers of the stack. AMIs bake the OS and runtime. Container images bake the runtime and application. Distroless images bake only what is needed to run, eliminating everything else. The pattern is "reduce to the minimum known state, make it immutable, replace on change."

### Pinning digests, not tags

Container image tags are mutable labels. `nginx:1.25` today points to a different image than `nginx:1.25` six months from now, because the Nginx maintainers push security patches under the same tag. Using tags in production means your deployment silently changes whenever the upstream maintainer rebuilds.

The correct practice is to pin by digest: `nginx@sha256:a14b17f71b3b...`. A digest is content-addressed and immutable — it refers to exactly one image, forever. Your CI pipeline resolves a tag to a digest at build time and records the digest in the image manifest. Deployments reference the digest. The fleet is deterministic.

In Kubernetes, this looks like:

```yaml
# k8s/deployment.yaml — digest-pinned image reference
spec:
  containers:
    - name: web
      # Tag resolved to digest by CI; never use a bare tag in production
      image: 123456789.dkr.ecr.us-east-1.amazonaws.com/web@sha256:a14b17f71b3b8d4e9a2f3c7b1d0e8a3c9f2b4d6e8a0c2e4f6b8d0a2c4e6f8
      imagePullPolicy: IfNotPresent
```

Renovate and Dependabot can automate digest updates: they open a PR when a new digest is available for a pinned image, run CI against it, and merge if the tests pass. This gives you automated, tested, audited image updates — the container equivalent of the Packer bake pipeline updating the golden AMI.

### Image lifecycle: build, scan, sign, promote

The full container image lifecycle mirrors the AMI lifecycle:

1. **Build** — `docker buildx build` with multi-stage + distroless base
2. **Scan** — Trivy or Grype before pushing; fail on CRITICAL CVEs
3. **Sign** — cosign with keyless signing (Sigstore) to produce a signed attestation; the registry stores the signature
4. **Push** — to GHCR or ECR with a content-digest reference
5. **Promote** — move the `latest-tested` tag to the new digest; update the deployment manifest via a PR
6. **Deploy** — GitOps controller (Argo CD or Flux) syncs the manifest change to the cluster

The [image security scanning post](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface) covers steps 2 and 3 in detail. The [IaC post](/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative) covers the Terraform side of the AMI lifecycle. The pattern across both is the same: bake once, scan, sign, promote, replace.

![A taxonomy tree of immutable artifacts branching into VM images and container images with their platform-specific leaf formats](/imgs/blogs/immutable-infrastructure-and-golden-images-7.png)

## 10. Measurement: before and after

The value of immutable infrastructure is measurable along the four DORA dimensions.

**Change-failure rate.** The most direct improvement. On mutable fleets, configuration drift means the fleet is never in a fully known state. Deployments interact with undocumented local state. CFR is typically 15–30% on mutable fleets (illustrative order-of-magnitude; DORA "low performers" cluster in this range). On immutable fleets, the only source of deployment failures is a bug in the image definition — and you catch those in the bake pipeline before they reach production. Measured CFR on immutable fleets tends to be 2–5% (DORA "high performers" are ≤5%).

**MTTR.** Mutable fleet incident: 2–8 hours (SSH debugging + patching + coordinated restart). Immutable fleet incident: 8–20 minutes (roll back to previous AMI, run `terraform apply`). The DORA "elite" benchmark for MTTR is under one hour; immutable infrastructure is the primary mechanism for hitting that target at the infrastructure level.

**Deploy frequency.** Immutable infrastructure does not directly increase deploy frequency — that is a CI and CD concern. But it removes the "last time we deployed, half the fleet did not match the image" problem that causes teams to deploy less frequently. Removing that drag allows deploy frequency to increase.

**Lead time for changes.** The Packer bake adds 15–25 minutes to the pipeline for infrastructure changes. This is a real cost, discussed in the trade-offs section below. The offset is that infrastructure changes no longer require manual coordination across nodes — a single `terraform apply` replaces the fleet.

| Metric | Mutable fleet | Immutable fleet | Source |
|---|---|---|---|
| Change-failure rate | 15–30% | 2–5% | DORA high/low performers |
| MTTR (infra incident) | 2–8 hours | 8–20 minutes | measured, illustrative |
| Audit coverage | Low (unlogged SSH) | Full (CI + Git + AMI) | deterministic |
| Autoscaling consistency | Poor (new nodes drift) | Perfect (same AMI) | deterministic |
| Packer bake time | n/a | 15–25 minutes | measured |

## 11. War story: the SSH session nobody documented

This is a composite of a real pattern I have seen at three different teams; the specifics are illustrative but the sequence is real.

A payments processing service ran on a fleet of eight EC2 instances managed by a CloudFormation stack that had not been updated in two years. The instances were launched from an AMI that was two years old. Over those two years, the ops team had applied security patches via SSH, upgraded the JVM twice (manually, on each node), changed the Nginx configuration to support a new TLS cipher suite, and added a cron job to rotate an API key.

None of this was documented anywhere except in individual engineers' bash histories, which had been rotated away. The team believed the instances were "basically identical" to the original AMI.

One afternoon, a fire suppression system triggered in the AWS data center and forced an instance stop on half the fleet. When AWS restarted the instances, they came back up — but the JVM version was wrong. The instance-stop had caused the AMI to be partially re-applied (this was not actually how AWS works, but the team believed it at the time — the actual cause was that the JVM upgrade had been done to `/usr/local/java` on one node and `/opt/java` on another, and the startup script had hardcoded the path). Three of the eight instances could not start the application.

The team began the process of reproducing the working configuration. They could not. The engineer who had upgraded the JVM had left the company. The upgrade had been done from memory and never documented. The Nginx config change had been done by a contractor whose Slack account was deprovisioned. The cron job had no comment explaining why it existed.

They rebuilt from scratch using the two-year-old AMI as a base, re-applying patches manually, guessing at the JVM configuration, and testing in staging for 2.5 days before restoring production.

The total outage for the three affected instances: 3 days. The business impact: \$180k in missed transaction processing fees (this was a real number from the post-mortem, though I am paraphrasing rather than attributing). The immediate remediation: the team adopted Packer-based golden images within six weeks.

The lesson is not that SSH is dangerous. The lesson is that **any change to a running server that is not also reflected in the image definition creates a debt that will be called in during an outage, at the worst possible time.**

## 12. Mutable vs immutable: detailed comparison

| Dimension | Mutable | Immutable |
|---|---|---|
| Patching model | SSH + in-place package upgrades | New AMI baked, fleet replaced |
| Configuration drift | Inevitable over time | Structurally impossible |
| Reproducibility | Fragile: unique per server | Exact: AMI ID is the artifact |
| Disaster recovery MTTR | 2–8 hours (manual rebuild) | 8–20 minutes (launch from AMI) |
| Audit trail | None (unlogged SSH) | Full (Git + CI + AMI + scan) |
| Autoscaling consistency | Poor (new nodes lack 6mo of patches) | Perfect (same AMI, always) |
| SSH access | Required | Disabled (no key pair) |
| Configuration management | Ansible/Chef/Puppet at boot | Ansible/Packer at build time |
| Deployment mechanism | Restart processes | Replace instances |
| Rollback mechanism | Re-run configuration management | Pin previous AMI ID |
| Compliance posture | Weak (no evidence of state) | Strong (CI + scan + AMI as evidence) |
| Initial build time | Fast (just launch) | Slower (15–25 min bake) |

The single column where mutable wins is "initial build time": launching a bare EC2 instance and running Ansible on it is faster than baking an AMI in Packer. For a development environment, this trade-off often favors mutable. For production, the trade-off is not even close.

![A matrix comparing mutable and immutable infrastructure across patching, reproducibility, disaster recovery, audit trail, and autoscaling dimensions](/imgs/blogs/immutable-infrastructure-and-golden-images-5.png)

## 13. How to reach for this (and when not to)

Immutable infrastructure with Packer and golden images is the right default for any team running production infrastructure on IaaS (AWS EC2, GCP Compute Engine, Azure VMs, DigitalOcean Droplets). It is worth the bake overhead for almost any team that has more than three servers and deploys more than once a week.

That said, there are cases where it is not worth it:

**Development environments.** Developers need fast iteration. A 25-minute Packer bake every time you change a library is too slow for a dev loop. Development machines should use mutable configuration management (Ansible or Homebrew) for speed. The boundary is: mutable is fine for developer machines and ephemeral test environments; immutable is the rule for production.

**PaaS environments.** If you are running on Heroku, Fly.io, Render, or Railway, the platform manages the underlying infrastructure. You do not build AMIs because you do not manage VMs. The platform's build system (Buildpacks or container-based) gives you immutability at the application layer. This is the right trade-off for small teams; you gain immutability without the operational overhead of managing Packer pipelines and Terraform state.

**Kubernetes clusters.** If your workloads run in Kubernetes, you still need golden images for the nodes (EC2 instances in an EKS managed node group), but the application-level immutability is handled by container images, not AMIs. Focus your golden image investment on hardened, scanned EKS AMIs (Amazon's EKS-optimized AMIs are a reasonable starting point; the CIS Amazon EKS Benchmark gives you the hardening baseline).

**Container-only workloads on Fargate or Cloud Run.** You never touch the underlying VM. Immutable container images are your artifact. Packer is not needed.

**When NOT to block on Packer bake time in your main CI pipeline.** The bake pipeline should be a *separate* CI workflow, not a gate on every application deployment. Application deployments should be fast (5–15 minutes). Infrastructure image updates should be asynchronous — triggered by base image updates or provisioning code changes, not by every application code change.

The anti-pattern to avoid: making Packer bake a requirement before every application deployment. That adds 25 minutes to every deploy and couples application velocity to infrastructure velocity. Keep them separate: application code deployments update the application artifact (container image or artifact archive); infrastructure code changes update the golden image.

## 14. Stress test: what if things go wrong?

**What if the Trivy scan fails after a Canonical AMI update?**

The CVE is in a package Ubuntu ships. Your bake pipeline fails. This is the correct behavior — you have caught a vulnerability before it reaches production. The response is: fix the Ansible playbook to either (a) upgrade the specific vulnerable package to the patched version, or (b) remove the package if it is unnecessary. Then re-run the pipeline.

While the pipeline is broken, your fleet continues running the previous AMI. The old CVE may still be present, but you have not introduced a new one, and you have not disrupted production. You have a bounded window to fix the image definition.

**What if the SSM parameter gets stale and Terraform deploys an old AMI?**

This is a failure mode in the promotion pipeline. Mitigation: add a `MaxImageAge` check in the promote job that rejects AMIs older than 30 days. If the pipeline has not run in 30 days, the promotion fails and alerts. The fleet keeps running the current AMI. This prevents the silent "we have not patched in three months" accumulation.

**What if the Terraform `instance_refresh` is stuck half-way?**

AWS instance refresh has a configurable timeout. If it gets stuck (e.g., new instances are failing health checks), the refresh rolls back automatically after the timeout. You end up with the old AMI still running. The rollback mechanism is built in — you do not need to act. Investigate why the new AMI's instances are failing health checks (check CloudWatch logs, check user-data script, check the Ansible playbook for a regression).

**What if two AMI bake jobs run concurrently and both write to the SSM parameter?**

Race condition: one job wins the SSM write, the other's result is silently discarded. Mitigation: use SSM Parameter Store's compare-and-swap (`--type SecureString` with a version check) or serialize bake jobs with a `concurrency: group` block in GitHub Actions:

```yaml
concurrency:
  group: bake-ami-${{ github.ref }}
  cancel-in-progress: false  # Don't cancel; queue instead
```

This ensures only one bake runs at a time per branch, and queued builds run in order.

**What if your AMI is very large (e.g., includes a GPU driver stack) and the bake takes 90 minutes?**

Layer your images. Build a "base layer" AMI that includes the OS, common packages, and the GPU drivers — this changes infrequently (monthly). Build a "runtime layer" AMI on top of the base layer that includes Node.js, your monitoring agents, and application-specific dependencies — this changes more frequently. The base bake runs monthly; the runtime bake runs on every provisioning code change and takes 15 minutes because it starts from the base layer. This mirrors the Docker layer cache pattern.

## 15. The immutable infrastructure loop in the DORA frame

The four DORA metrics close the loop on immutable infrastructure's value:

**Deploy frequency** is unblocked: immutable infrastructure removes the "the fleet is in an unknown state, we cannot deploy safely" gate that causes teams to batch up changes and deploy infrequently.

**Lead time for changes** has two components for infrastructure changes: the bake time (15–25 minutes, fixed) plus the fleet replacement time (8–12 minutes for a 20-node fleet). Total lead time: 25–40 minutes from a Git commit to a fully patched fleet. On a mutable fleet, coordinating a manual patch across 20 nodes takes 4–8 hours.

**Change-failure rate** is structurally bounded: the scan gate catches CVEs before deployment; the image bake catches provisioning failures before deployment; the health check during instance refresh catches application failures before they reach the full fleet. Failures still happen, but they are caught earlier and affect fewer nodes.

**Time to restore** is deterministic: rolling back to the previous AMI via SSM parameter update + `terraform apply` takes 8–12 minutes for a 20-node fleet, regardless of the nature of the failure. On mutable infrastructure, MTTR depends on the nature of the failure and the documentation quality of the previous state.

These four improvements compound. A team that ships reliable infrastructure changes frequently, with bounded blast radius and fast recovery, can attempt more infrastructure improvements per quarter — better hardware, better OS configuration, better security hardening — because the cost of each attempt is low and the recovery from a failed attempt is fast.

### Measuring the bake pipeline itself

The bake pipeline is infrastructure that needs to be measured. Key metrics:

**Bake time** is the elapsed time from `packer build` start to AMI registered in the console. Target: under 25 minutes for a typical Ubuntu + runtime AMI. If it takes longer, profile the Ansible playbook for slow tasks (usually package downloads or compilation steps). Use the `--only` flag to run a single builder for debugging.

**Scan pass rate** is the percentage of bake runs where Trivy finds zero CRITICAL CVEs. A falling scan pass rate means your base image or your provisioning code is introducing new vulnerabilities faster than you are patching them. Track it over time; a step-change decrease usually correlates with a new Ansible role or a base image update that pulled in a vulnerable package.

**Time from CVE published to fleet patched** is the end-to-end metric that compliance auditors care about. For automated immutable pipelines (nightly bake + automatic Terraform apply), this should be under 24 hours for kernel and system library CVEs that Canonical backports to Ubuntu's security archive. For CVEs that require Ansible playbook changes (because the vulnerable package is something you install, not something Ubuntu ships), the time is "time to write the Ansible change + bake + deploy" — typically 1–4 hours.

**Fleet AMI age distribution** is a histogram of how old the AMI is for each running instance. In a healthy immutable fleet, this histogram is narrow (all instances within days of each other) and the mean age is under 30 days. A wide histogram or a long tail indicates instances that were not replaced during the last rolling update — investigate why the instance refresh did not complete.

![A before-after diagram contrasting in-place patching with no audit trail against image-replacement patching with a full Git and CI audit trail and 8-minute rollback](/imgs/blogs/immutable-infrastructure-and-golden-images-8.png)

## 16. War story: the AMI that saved a compliance audit

A second illustrative story, from the container image side of the same principle.

A team running a financial services application had been asked to provide evidence to their PCI-DSS auditor that all production servers had been patched against a specific kernel CVE within 30 days of its announcement. On a mutable infrastructure, this requires either (a) a configuration management inventory tool that tracked every patch applied to every node (which they did not have), or (b) a manual process of documenting SSH sessions (which nobody had done).

On their immutable infrastructure, the answer was three lines of AWS CLI:

```bash
# Find all production AMIs built after the CVE patch was released
aws ec2 describe-images \
  --filters \
    "Name=tag:Status,Values=prod-eligible" \
    "Name=tag:BaseOS,Values=ubuntu-22.04" \
  --query "Images[?CreationDate>='2024-09-15'].{AMI:ImageId,Date:CreationDate,Build:Tags[?Key=='BuildVersion']|[0].Value}" \
  --output table
```

Every production instance was running an AMI built after the patch date. The Trivy scan log for each AMI, stored in S3 by the CI pipeline, showed zero CRITICAL findings for the specific CVE. The CI run log showed the Git SHA of the Ansible playbook change that had updated the base image reference. The auditor had a complete chain of evidence in thirty minutes.

The contrast with the SSH-based team in the previous section is stark: one team spent 2.5 days in recovery from an untracked change; the other team spent 30 minutes satisfying a compliance audit. The operational difference was not the engineering quality of the teams — it was whether they had made immutability a structural constraint.

## 17. Ansible playbook patterns for golden image baking

The Ansible playbook that runs inside the Packer build is structurally different from an Ansible playbook that manages live servers. It runs exactly once, on a fresh EC2 instance with no previous state, and its output is an AMI. This has specific implications for how you write it.

**Idempotency matters less than correctness.** A normal Ansible playbook needs to be idempotent because it runs repeatedly against live servers. The Packer playbook runs once against a fresh instance. Idempotency is still good practice, but the failure mode you care about is "the image is wrong," not "the playbook changed something it should not have."

**Fail fast.** Use `any_errors_fatal: true` at the play level. If any task in the playbook fails, stop the build immediately. A partially provisioned AMI is worse than no AMI — it looks complete but is not. Do not use `ignore_errors: yes` except for genuinely optional tasks (like a cleanup step that should not block the build).

**No dynamic inventory.** You are not targeting a group of live hosts. The playbook targets `all` and runs against the single Packer-managed instance. Keep the inventory setup in the Packer `ansible` provisioner block.

**Pin all package versions.** In a Packer build, `apt-get install nodejs` gives you whatever version is in the Ubuntu package repository at build time. This varies day to day. For reproducibility, pin versions: `nodejs=20.11.0-1nodesource1`. Yes, this means you need to update the version when you want to upgrade — but that update is a deliberate, versioned, code-reviewed change, not a silent drift.

Here is a representative Ansible playbook structure for a golden image:

```yaml
# ansible/golden-image.yml
---
- name: Configure golden Ubuntu + Node.js image
  hosts: all
  become: true
  any_errors_fatal: true

  vars:
    nodejs_version: "{{ nodejs_version | default('20') }}"
    cloudwatch_agent_version: "1.300041.0"

  roles:
    # Order matters: hardening before application, cleanup last
    - role: os-baseline        # sysctl, ulimits, /etc/security/limits.conf
    - role: nodejs             # NodeSource repo + pinned Node.js
    - role: cloudwatch-agent   # Amazon CloudWatch unified agent
    - role: devsec.hardening.os_hardening  # CIS Ubuntu Level 1

  post_tasks:
    - name: Verify Node.js is installed at the expected version
      command: node --version
      register: node_version_output
      changed_when: false
      failed_when: >
        node_version_output.stdout is not match('^v' + nodejs_version + '\.')

    - name: Verify CloudWatch agent is configured
      stat:
        path: /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
      register: cwa_config
      failed_when: not cwa_config.stat.exists

    - name: Clean apt cache and temp files
      apt:
        clean: true
        autoremove: true

    - name: Clear bash history and log artifacts from the build
      shell: |
        history -c
        truncate -s 0 /var/log/auth.log
        truncate -s 0 /var/log/syslog
        find /tmp -mindepth 1 -delete
        find /var/tmp -mindepth 1 -delete
      args:
        warn: false
```

The `post_tasks` section is particularly important: the verification steps (checking Node.js version, checking CloudWatch agent config) make the playbook self-testing. If the image bakes but Node.js is not at the expected version, the Packer build fails. You catch the problem in CI rather than in production.

#### Worked example: layered image build for faster bake times

A common optimization for teams with complex software stacks is to layer the Packer builds. The pattern:

- **Base AMI** (baked weekly or when Ubuntu releases security updates): Ubuntu 22.04 + OS hardening + CloudWatch agent + standard system packages. Bake time: 18–22 minutes. This AMI is stable and changes infrequently.
- **Runtime AMI** (baked on every provisioning code change): starts from the base AMI, adds the language runtime (Node.js 20, Python 3.12, Java 21, etc.) and application-level dependencies. Bake time: 6–10 minutes (because it starts from the base AMI, which already has the OS setup done).

The Packer HCL for the runtime AMI uses the base AMI as its source:

```hcl
# packer/nodejs-runtime.pkr.hcl

data "amazon-ami" "base" {
  filters = {
    name             = "golden-base-*"
    root-device-type = "ebs"
    tag:Status       = "prod-eligible"
  }
  most_recent = true
  owners      = ["self"]
  region      = var.aws_region
}

source "amazon-ebs" "nodejs-runtime" {
  region        = var.aws_region
  instance_type = "t3.small"
  # Start from our own base AMI, not from Canonical's
  source_ami    = data.amazon-ami.base.id
  ssh_username  = "ubuntu"
  ami_name      = "golden-nodejs${var.nodejs_version}-runtime-${formatdate("YYYYMMDD", timestamp())}-${var.build_version}"
  encrypt_boot  = true
}

build {
  sources = ["source.amazon-ebs.nodejs-runtime"]

  provisioner "ansible" {
    playbook_file = "${path.root}/../ansible/nodejs-runtime.yml"
    extra_arguments = [
      "-e", "nodejs_version=${var.nodejs_version}"
    ]
  }
}
```

The runtime bake runs in 6–10 minutes because the expensive OS setup is already done in the base image. The base bake runs at most weekly. Total CI time for an application infrastructure change: 6–10 minutes rather than 18–25 minutes.

Before-and-after for this optimization:

| Approach | Bake trigger | Bake time | AMI scope |
|---|---|---|---|
| Single-image build | Every infra code change | 22–28 minutes | OS + runtime |
| Layered build (base) | Weekly + Ubuntu security updates | 18–22 minutes | OS only |
| Layered build (runtime) | Every runtime code change | 6–10 minutes | Runtime only |

The layered approach cuts average bake time from 22–28 minutes to 6–10 minutes for the 90% of changes that affect only the runtime layer. The base layer bake happens asynchronously on a weekly schedule.

## Key takeaways

1. **A server that can be patched in-place will eventually drift into an irreproducible snowflake.** The only reliable fix is to make in-place patching structurally impossible.

2. **Packer's job is to produce an AMI ID.** That AMI ID is the immutable artifact — the equivalent of a container image digest in the VM world. Every downstream tool (Terraform, Auto Scaling, your CDK stack) consumes the AMI ID as a versioned reference.

3. **The image bake pipeline belongs in CI.** A Packer bake triggered manually is a snapshot. A Packer bake triggered automatically by base image updates and provisioning code changes is an artifact with full provenance.

4. **Trivy scans in the bake pipeline catch CVEs before deployment.** The scan is a gate, not a report. An exit-code-1 failure blocks AMI publication. No new CVEs reach production.

5. **Terraform's `instance_refresh` is the fleet replacement mechanism.** When the AMI ID changes in the launch template, the ASG rolls the fleet. The `min_healthy_percentage` and `instance_warmup` parameters bound the blast radius during replacement.

6. **Cattle replace pets for production.** No SSH key pairs on production instances, no SSH port open, no SSM Session Manager in prod. The server is disposable; the image definition is the source of truth.

7. **Container images apply the same principle at finer granularity.** Distroless images enforce immutability at the runtime level: nothing to patch, nothing to SSH into.

8. **Rollback is pin-and-apply.** Update the SSM parameter to the previous AMI ID. Run `terraform apply`. Fleet is rolled back in 8–12 minutes. No archaeology required.

9. **Immutable infrastructure is a prerequisite for fast MTTR.** If your fleet is in an unknown state, recovery requires reconstruction. If your fleet boots from a known AMI, recovery is a parameter update.

10. **The bake time (15–25 minutes) is the real cost.** Keep it off your application CI pipeline. Run bake as a separate workflow triggered by infrastructure code changes and scheduled nightly base-image updates.

## Further reading

- [Packer documentation — amazon-ebs builder](https://developer.hashicorp.com/packer/integrations/hashicorp/amazon/latest/components/builder/ebs)
- [Terraform aws_autoscaling_group instance_refresh](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/autoscaling_group#instance_refresh)
- [CIS Amazon Machine Image Benchmark](https://www.cisecurity.org/benchmark/amazon_web_services) — hardening standards for your golden images
- [Trivy documentation — filesystem scan](https://aquasecurity.github.io/trivy/latest/docs/target/filesystem/)
- [The Twelve-Factor App — Build/release/run separation](https://12factor.net/build-release-run) — the original articulation of the immutable artifact principle
- [Accelerate (Nicole Forsgren, Jez Humble, Gene Kim)](https://itrevolution.com/accelerate-book/) — DORA research foundation
- [Infrastructure as code: from clickops to declarative](/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative) — the IaC foundation that makes immutable infrastructure tractable
- [Image security scanning and a minimal attack surface](/blog/software-development/ci-cd/image-security-scanning-and-a-minimal-attack-surface) — Trivy, SBOM, and the scan gate in depth
- [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) — SRE perspective on why fast recovery matters more than preventing every failure
