---
title: "IaC for the whole stack and config drift: DNS, IAM, and beyond"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn to version-control every layer of your stack — DNS, IAM, managed databases, CDN, and monitoring — and detect and eliminate the configuration drift that turns routine Terraform applies into production outages."
tags:
  [
    "ci-cd",
    "devops",
    "infrastructure-as-code",
    "terraform",
    "drift-detection",
    "iam",
    "configuration-management",
    "gitops",
    "platform-engineering",
    "sre",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/iac-for-the-whole-stack-and-config-drift-1.png"
---

It was a Tuesday afternoon deploy. The kind that had gone smoothly twenty times in a row, so no one was watching closely. The Terraform `apply` ran in the pipeline, the plan summary showed eight changes — mostly updating autoscaling group tags and a Lambda function. Then the alerts started. The payment service was returning 403 Forbidden. The on-call engineer opened CloudWatch, then the IAM console, and found it immediately: the `lambda-payment-processor` role was missing the policy that allowed it to call the Secrets Manager endpoint. The policy was gone. Not revoked — gone, as if it had never existed. Except it had existed, for four months, added by a developer who needed a quick fix one evening and figured they'd "add it to Terraform later." Later never came. The Terraform state never knew about it. So when `apply` ran, Terraform saw a resource it didn't own and quietly removed it, right alongside the eight legitimate changes. Forty-seven minutes of outage, a postmortem, and a very uncomfortable conversation with the CTO.

This is the canonical IaC drift story. Not a dramatic attack or a botched migration — just a well-intentioned one-liner in the AWS console that the pipeline eventually cleaned up on behalf of no one's intentions. The problem is not that Terraform did something wrong. Terraform did exactly what it was told: reconcile reality with state. The problem is that IAM was only half in Terraform, which is the same as being zero in Terraform when `apply` runs.

Most teams start their IaC journey with compute — EC2 instances, autoscaling groups, maybe Lambda. That is reasonable: compute is what you deploy your code to, so it is the obvious first target. But a production stack is not just compute. It is also DNS records, TLS certificates, CDN distributions, load balancer rules, IAM roles and policies, managed database parameter groups and backup windows, security group rules, monitoring dashboards, and alert rules. Every one of these resources that lives outside your Terraform state is a time bomb. It will either drift silently and cause a surprise on the next `apply`, or it will be changed by someone in a rush and disappear from institutional memory.

By the end of this post you will know how to extend IaC coverage to the entire stack — Route 53, ACM, CloudFront, RDS, IAM — with real Terraform HCL you can adapt today. You will understand what drift is, why it accumulates faster than any team expects, and how to run automated drift detection in CI so you find out about changes before they bite you on apply. You will have a reconciliation playbook: what to do when drift is discovered, with the three options (update the code, re-apply to remove it, or `terraform import` to adopt it), and when to reach for each one. And you will see the maturity model that describes where most teams are and where whole-stack IaC coverage actually lives. This post is part of the broader [CI/CD mental model series](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the "everything as code" principle extends far beyond pipelines and Dockerfiles.

![All seven layers of a production stack from DNS through monitoring all managed as code in version control](/imgs/blogs/iac-for-the-whole-stack-and-config-drift-1.png)

---

## 1. Why IaC coverage stops at compute — and why that is a trap

Teams do not leave IAM and DNS out of Terraform because they are lazy. They leave them out because those resources feel different. IAM feels like a security concern, something you coordinate with a security team and change carefully one at a time. DNS feels like infrastructure-of-infrastructure, something that was set up once years ago and should not be touched. Managed databases feel risky to codify because what if Terraform decides to recreate the instance? Monitoring dashboards feel optional — they are Grafana JSON that engineers usually click into existence.

All of these feelings are rational, and all of them lead to the same outcome: a Terraform state that covers 40% of your actual infrastructure, with the other 60% in a combination of console click-history, tribal knowledge, and runbooks that were last updated in 2021.

The gap creates a specific failure pattern. Your `terraform plan` for a routine change shows eight additions and one modification. Four of those additions are unrelated to the change you intended — they are Terraform reconciling differences it noticed between state and reality on resources you share a module with. You review them quickly, assume they are fine, and apply. One of them was not fine. It removed an IAM permission that some service had been relying on for three months, added via a console that none of the Terraform code knew about.

The math of this failure is worth making explicit. If your IaC coverage is $C$ (the fraction of your infrastructure that lives in Terraform state) and your team makes $N$ manual changes per month, then the expected number of surprise-apply incidents per month is roughly proportional to $N \times (1 - C)$. Double your coverage, halve your surprises. Get to 95% coverage, and the residual is close to zero. The goal is not 100% — there are always some resources that are genuinely too risky to import or manage programmatically. The goal is to know exactly which resources those are and to never accidentally apply against them.

The fix is not to stop using the console for IAM. The fix is to make the console change the exception, not the rule. That means codifying IAM from the start, building a no-console-in-prod policy that enforces read-only access for everyone except a break-glass emergency role, and treating every console change as a debt you pay off before the next `apply` runs.

### The hidden cost of partial IaC coverage

Partial IaC coverage does not just create drift risk — it creates a specific cognitive hazard. When engineers know that the Terraform state is not authoritative, they stop trusting `terraform plan` output. They see a diff of twelve changes and cannot tell which six are expected and which six are surprises. So they either apply blindly (dangerous) or hesitate to apply at all (the change never gets deployed). A team that hesitates to run `terraform apply` because they are afraid of surprises is a team that will start making more console changes — because at least those are predictable. This is the spiral: partial coverage generates distrust, distrust generates manual changes, manual changes generate more partial coverage. The spiral ends at full IaC coverage or at a catastrophic incident.

There is a measurable signal for how deep in the spiral a team is: the average number of unexpected changes in a `terraform plan`. A team with full coverage sees zero unexpected changes on a routine run — every item in the plan was authored intentionally. A team partway through the spiral sees three to eight unexpected changes per run. A team deep in the spiral has stopped running `terraform plan` at all and is applying on a wing and a prayer.

The other measurable signal is the "last applied" timestamp in the Terraform remote state. Teams that are afraid of their own Terraform state tend to apply infrequently. Teams that trust their Terraform state apply frequently — because they know each apply is small, predictable, and reversible if something goes wrong. The path from infrequent, scary applies to frequent, boring ones runs directly through IaC coverage. This is the same insight that underpins the DORA deploy frequency metric: the teams with the highest deploy frequency are the ones with the most confidence in their pipeline, and confidence comes from coverage and drift detection.

---

## 2. DNS and TLS as code: Route 53 + ACM

DNS is the most underrated piece of infrastructure to put in Terraform, because DNS failures are some of the most spectacular and hard-to-debug outages. A missing A record means the service is unreachable. A wrong CNAME after a migration means the old service gets traffic. A certificate that expired because no one automated renewal means every browser shows a scary warning. All of these are operationally devastating, and all of them are entirely preventable when DNS and TLS are code.

The mental shift is simple: treat every domain name and every certificate the same way you treat an EC2 instance. It is a resource with a lifecycle. It must be created, updated, and eventually deleted. The create-update-delete lifecycle must be tracked. And since DNS changes propagate globally in minutes, an untested change can break a production service for thousands of users before anyone notices.

Here is a real Terraform module that provisions a hosted zone, an apex A record pointing to a CloudFront distribution, and an ACM certificate with DNS validation:

```hcl
# modules/dns-tls/main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "domain_name" {
  description = "Root domain (e.g., example.com)"
  type        = string
}

variable "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  type        = string
}

variable "cloudfront_zone_id" {
  description = "CloudFront hosted zone ID (fixed: Z2FDTNDATAQYW2)"
  type        = string
  default     = "Z2FDTNDATAQYW2"
}

variable "tags" {
  type    = map(string)
  default = {}
}

# Hosted zone — the DNS namespace for your domain
resource "aws_route53_zone" "main" {
  name = var.domain_name
  tags = var.tags
}

# ACM certificate — created in us-east-1 because CloudFront requires it
resource "aws_acm_certificate" "main" {
  provider          = aws.us_east_1
  domain_name       = var.domain_name
  subject_alternative_names = ["*.${var.domain_name}"]
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = var.tags
}

# DNS validation record — Terraform creates the CNAME that ACM checks
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.main.zone_id
}

# Wait for certificate to be validated before proceeding
resource "aws_acm_certificate_validation" "main" {
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.main.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}

# Apex A record pointing to CloudFront
resource "aws_route53_record" "apex" {
  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = var.cloudfront_domain_name
    zone_id                = var.cloudfront_zone_id
    evaluate_target_health = false
  }
}

# www CNAME pointing to apex
resource "aws_route53_record" "www" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.${var.domain_name}"
  type    = "CNAME"
  ttl     = 300
  records = [var.domain_name]
}

output "zone_id" {
  value = aws_route53_zone.main.zone_id
}

output "certificate_arn" {
  value = aws_acm_certificate_validation.main.certificate_arn
}
```

Notice the `lifecycle { create_before_destroy = true }` on the certificate. Without this, Terraform would delete the old certificate before creating the new one, which would briefly break HTTPS for your entire site. The `create_before_destroy` lifecycle rule is not optional here — it is the difference between a smooth cert rotation and a production incident.

The DNS validation records are provisioned automatically using `for_each` over the `domain_validation_options` output from ACM. This means your certificate validation is handled entirely by Terraform: no manual step where someone has to copy-paste a CNAME into Route 53. Automated, reproducible, and visible in a `plan` before it happens.

---

## 3. IAM as code: the most dangerous resource to hand-manage

IAM is the most important and the most dangerous resource to manage manually. IAM controls who and what can do what to your infrastructure. A missing IAM permission breaks your application. An extra IAM permission creates a security vulnerability. Both of these are changed silently in the console by developers who need something to work quickly, and both of them create drift that Terraform will enthusiastically reconcile on the next apply — possibly in the wrong direction.

There are two distinct IAM drift scenarios:

**The deletion scenario**: someone adds a permission via the console. Terraform doesn't know about it. Next `apply`, Terraform removes it. Service breaks.

**The inflation scenario**: someone adds a permission via the console. Terraform doesn't know about it. Six months later, a security audit finds over-privileged roles. You try to lock down IAM via Terraform, but the console permission is now load-bearing and you don't know it. You remove it in the audit, service breaks.

Both scenarios share the same root cause: the console was used to modify something that Terraform was supposed to own. The fix is not to audit better — it is to make console modification structurally impossible for most users.

Here is a Terraform module for a minimal but real IAM role setup — an ECS task role with least-privilege permissions:

```hcl
# modules/iam-ecs-role/main.tf

variable "service_name" {
  description = "Name of the ECS service this role serves"
  type        = string
}

variable "secrets_arns" {
  description = "List of Secrets Manager secret ARNs the service needs"
  type        = list(string)
  default     = []
}

variable "s3_bucket_arns" {
  description = "List of S3 bucket ARNs the service can read from"
  type        = list(string)
  default     = []
}

variable "tags" {
  type    = map(string)
  default = {}
}

# The trust policy — only ECS tasks can assume this role
data "aws_iam_policy_document" "assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task" {
  name               = "${var.service_name}-task-role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
  tags               = var.tags
}

# Secrets Manager — read-only access to specific secrets
data "aws_iam_policy_document" "secrets" {
  count = length(var.secrets_arns) > 0 ? 1 : 0

  statement {
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = var.secrets_arns
  }
}

resource "aws_iam_policy" "secrets" {
  count  = length(var.secrets_arns) > 0 ? 1 : 0
  name   = "${var.service_name}-secrets-policy"
  policy = data.aws_iam_policy_document.secrets[0].json
  tags   = var.tags
}

resource "aws_iam_role_policy_attachment" "secrets" {
  count      = length(var.secrets_arns) > 0 ? 1 : 0
  role       = aws_iam_role.task.name
  policy_arn = aws_iam_policy.secrets[0].arn
}

# S3 — read-only access to specific buckets
data "aws_iam_policy_document" "s3" {
  count = length(var.s3_bucket_arns) > 0 ? 1 : 0

  statement {
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = concat(
      var.s3_bucket_arns,
      [for arn in var.s3_bucket_arns : "${arn}/*"]
    )
  }
}

resource "aws_iam_policy" "s3" {
  count  = length(var.s3_bucket_arns) > 0 ? 1 : 0
  name   = "${var.service_name}-s3-policy"
  policy = data.aws_iam_policy_document.s3[0].json
  tags   = var.tags
}

resource "aws_iam_role_policy_attachment" "s3" {
  count      = length(var.s3_bucket_arns) > 0 ? 1 : 0
  role       = aws_iam_role.task.name
  policy_arn = aws_iam_policy.s3[0].arn
}

output "role_arn" {
  value = aws_iam_role.task.arn
}

output "role_name" {
  value = aws_iam_role.task.name
}
```

The critical discipline here is resource naming. Every IAM resource gets a name derived from the service it belongs to. This makes it immediately obvious in the AWS console which team owns which role, and it makes `terraform import` straightforward when you need to adopt an existing role. It also makes the security audit easy: search for roles not matching the naming pattern and you have found your unmanaged IAM.

The organizational control that enforces this is a Service Control Policy (SCP) at the AWS Organizations level:

```hcl
# modules/scp/main.tf — SCP to enforce IaC-only IAM changes

data "aws_iam_policy_document" "deny_console_iam" {
  statement {
    sid    = "DenyConsoleIAMWrite"
    effect = "Deny"

    # Deny all IAM write actions
    actions = [
      "iam:AttachRolePolicy",
      "iam:CreatePolicy",
      "iam:CreateRole",
      "iam:DeletePolicy",
      "iam:DeleteRole",
      "iam:DeleteRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PutRolePolicy",
      "iam:UpdateAssumeRolePolicy",
    ]

    resources = ["*"]

    # Except for the Terraform pipeline role
    condition {
      test     = "StringNotLike"
      variable = "aws:PrincipalArn"
      values   = [
        "arn:aws:iam::*:role/terraform-pipeline-role",
        "arn:aws:iam::*:role/OrganizationAccountAccessRole",
      ]
    }
  }
}

resource "aws_organizations_policy" "deny_console_iam" {
  name        = "deny-console-iam-write"
  description = "Prevent all IAM writes except via Terraform pipeline role"
  content     = data.aws_iam_policy_document.deny_console_iam.json
  type        = "SERVICE_CONTROL_POLICY"
}
```

This SCP makes it structurally impossible for a developer to add an IAM policy via the console. It is the enforcement mechanism for the no-console-in-prod policy. Combined with the Terraform module above, IAM changes require a PR, a plan, a review, and an apply — exactly the same process as a code change.

![Before and after view showing manual IAM console changes causing Terraform deletion versus IaC-managed IAM producing clean reviewed applies](/imgs/blogs/iac-for-the-whole-stack-and-config-drift-2.png)

---

## 4. Managed databases, CDN, and load balancers as code

The three resource types that teams most commonly leave half-managed are RDS parameter groups, CloudFront distributions, and Application Load Balancer listener rules. Each one has the same failure pattern: the main resource is in Terraform (the RDS instance, the CloudFront distribution, the ALB), but the configuration details are managed via the console after the fact.

### RDS parameter groups and backup windows

When Terraform creates an RDS instance without an explicit `parameter_group_name`, it uses the default parameter group. The default parameter group cannot be modified. So when a DBA needs to tune `max_connections` or enable `slow_query_log`, they create a custom parameter group via the console, attach it to the instance, and now you have configuration living outside your state. Next time someone modifies the RDS Terraform resource, Terraform's plan will show the parameter group reverted to default.

```hcl
# modules/rds/main.tf

variable "identifier" { type = string }
variable "engine_version" { type = string }
variable "instance_class" { type = string }
variable "allocated_storage" { type = number }
variable "db_name" { type = string }
variable "username" { type = string }
variable "password" {
  type      = string
  sensitive = true
}
variable "subnet_ids" { type = list(string) }
variable "vpc_security_group_ids" { type = list(string) }
variable "backup_retention_days" {
  type    = number
  default = 7
}
variable "tags" { type = map(string); default = {} }

# Custom parameter group — tune here, not in the console
resource "aws_db_parameter_group" "main" {
  name   = "${var.identifier}-pg"
  family = "mysql8.0"

  parameter {
    name  = "max_connections"
    value = "500"
  }

  parameter {
    name  = "slow_query_log"
    value = "1"
  }

  parameter {
    name         = "long_query_time"
    value        = "2"
    apply_method = "immediate"
  }

  tags = var.tags
}

# Subnet group
resource "aws_db_subnet_group" "main" {
  name       = "${var.identifier}-subnet-group"
  subnet_ids = var.subnet_ids
  tags       = var.tags
}

resource "aws_db_instance" "main" {
  identifier     = var.identifier
  engine         = "mysql"
  engine_version = var.engine_version
  instance_class = var.instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.allocated_storage * 2  # auto-scaling ceiling
  storage_encrypted     = true

  db_name  = var.db_name
  username = var.username
  password = var.password

  # Use our managed parameter group, not the default
  parameter_group_name = aws_db_parameter_group.main.name

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = var.vpc_security_group_ids

  backup_retention_period = var.backup_retention_days
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.identifier}-final-snapshot"

  tags = var.tags
}
```

The `deletion_protection = true` attribute is the safety net that prevents Terraform from dropping the production database if someone accidentally removes the resource block. Combined with a Sentinel or OPA policy that enforces `deletion_protection` on all RDS instances, you have a belt-and-suspenders guard.

A practical concern teams always raise: "what if Terraform recreates the database when we change the parameter group?" This is a legitimate fear. The answer is that most parameter group changes do not require recreation — they apply in-place, sometimes immediately and sometimes after a maintenance window reboot. Terraform will tell you in the plan: look for `(requires new resource)` next to any attribute. If you see it, that is a signal to change your approach — use the `apply_method = "pending-reboot"` attribute in the parameter block, or use a maintenance window, or split the change from the instance modification. But the answer is never to avoid managing the parameter group in Terraform — the drift risk from console-managed parameters is worse than the operational complexity of safe parameter changes.

The `max_allocated_storage` attribute (set to twice the initial allocation) enables RDS autoscaling for storage. Without it, a storage-full event triggers an alert and requires a manual instance modification at 2 AM. With it, RDS expands automatically up to the ceiling. The ceiling is set in Terraform, which means the policy decision ("we are willing to pay up to 2x storage costs before we need to resize") is in a PR and a commit, not in someone's memory.

### CloudFront distribution as code

CloudFront is a particularly common drift source because distributions have many settings — cache behaviors, geo-restrictions, custom headers, WAF associations — and engineers frequently adjust these via the console to debug performance issues or implement quick security fixes.

```hcl
# modules/cloudfront/main.tf

variable "domain_name" { type = string }
variable "certificate_arn" { type = string }
variable "origin_domain_name" { type = string }
variable "web_acl_id" { type = string; default = null }
variable "tags" { type = map(string); default = {} }

resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  aliases             = [var.domain_name, "www.${var.domain_name}"]
  web_acl_id          = var.web_acl_id
  price_class         = "PriceClass_100"  # US/EU only — change in code, not console

  origin {
    domain_name = var.origin_domain_name
    origin_id   = "primary"

    custom_header {
      name  = "X-Origin-Verify"
      value = "cloudfront-only"
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "primary"
    compress         = true

    forwarded_values {
      query_string = true
      cookies { forward = "none" }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 86400
    max_ttl                = 31536000
  }

  # Static assets — long TTL, no query strings forwarded
  ordered_cache_behavior {
    path_pattern     = "/static/*"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "primary"
    compress         = true

    forwarded_values {
      query_string = false
      cookies { forward = "none" }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 2592000   # 30 days
    max_ttl                = 31536000  # 1 year
  }

  restrictions {
    geo_restriction { restriction_type = "none" }
  }

  viewer_certificate {
    acm_certificate_arn      = var.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = var.tags
}

output "domain_name" {
  value = aws_cloudfront_distribution.main.domain_name
}

output "hosted_zone_id" {
  value = aws_cloudfront_distribution.main.hosted_zone_id
}
```

Calling `price_class = "PriceClass_100"` directly in the code means the decision to serve only from US/EU edge locations is in a PR, not buried in the CloudFront console. When someone wants to expand to Asia-Pacific, they open a PR that changes this to `PriceClass_All` — it goes through review, gets discussed, and the business rationale (cost vs latency for APAC users) is documented in the PR description forever.

The `X-Origin-Verify` custom header in the origin block is a security pattern worth noting. CloudFront adds this header to every request it forwards to the origin. The origin (ALB or server) is configured to reject requests that do not carry this header. This means no one can bypass CloudFront and hit the origin directly — which matters if you are relying on CloudFront for WAF rules, DDoS protection, or geo-blocking. Managing this header in Terraform means the value is consistent between CloudFront and the origin's configuration, and rotating it requires a simultaneous PR to both resources. Without Terraform, rotating this header means two separate console operations with a gap between them where CloudFront is sending the old header and the origin rejects requests.

Load balancer listener rules are another common drift source. Teams create ALB path-based routing rules via the console to route traffic to a new service, then forget to codify them. Here is a minimal Terraform block for an ALB listener with two routing rules:

```hcl
# modules/alb/listeners.tf

resource "aws_alb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.default.arn
  }
}

# API traffic — distinct path prefix gets its own target group
resource "aws_alb_listener_rule" "api" {
  listener_arn = aws_alb_listener.https.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}

# Health check endpoint — returns 200 without touching the app
resource "aws_alb_listener_rule" "health" {
  listener_arn = aws_alb_listener.https.arn
  priority     = 10

  action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "OK"
      status_code  = "200"
    }
  }

  condition {
    path_pattern {
      values = ["/health"]
    }
  }
}
```

Priority 10 for the health check means it evaluates before the API rule (priority 100) and before the default forward. Every listener rule created via the console appears as a surprise on the next `terraform plan`. By keeping listener rules in Terraform, you also get an authoritative list of what traffic patterns your ALB handles — invaluable during incident response when you need to understand routing.

---

## 5. Monitoring and alerting as code

Monitoring is the IaC frontier that almost no team reaches before a painful incident. The drift pattern is familiar: an engineer creates a Grafana dashboard via the UI during an incident to understand a problem, and the dashboard saves. A week later it is load-bearing — it is the dashboard in the runbook, the one the on-call uses. But it lives in Grafana's database, backed up only if your Grafana backup process works, and invisible to Terraform or Git. When the Grafana instance is migrated or rebuilt, the dashboard is gone.

The same pattern applies to Prometheus alert rules, PagerDuty escalation policies, and CloudWatch alarms.

Grafana's Terraform provider lets you manage dashboards and alert rules as code:

```hcl
# modules/monitoring/main.tf

terraform {
  required_providers {
    grafana = {
      source  = "grafana/grafana"
      version = "~> 2.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "service_name" { type = string }
variable "alert_webhook_url" { type = string; sensitive = true }

# CloudWatch alarm — latency threshold
resource "aws_cloudwatch_metric_alarm" "latency_p99" {
  alarm_name          = "${var.service_name}-latency-p99-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "3"
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = "60"
  statistic           = "p99"
  threshold           = "2.0"
  alarm_description   = "p99 latency exceeds 2s for 3 consecutive minutes"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = "${var.service_name}-alb"
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
}

# SNS topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.service_name}-alerts"
}

resource "aws_sns_topic_subscription" "webhook" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "https"
  endpoint  = var.alert_webhook_url
}

# Grafana dashboard — the JSON content is stored in a file in the repo
resource "grafana_dashboard" "service" {
  config_json = file("${path.module}/dashboards/${var.service_name}.json")
  overwrite   = true
}
```

The `file("${path.module}/dashboards/${var.service_name}.json")` reference is the key pattern. The Grafana dashboard JSON lives in the repository, not in Grafana's database. Exporting it is a one-time step: click the dashboard in Grafana UI, select "Export as JSON", save to the `dashboards/` folder. From that point, the dashboard is code — versioned, reviewed, and deployable to any Grafana instance.

For Prometheus alert rules (if you are running Prometheus rather than CloudWatch), the code approach is a YAML file in the repository that the Prometheus operator watches:

```yaml
# infra/monitoring/rules/web-service.yaml
# Deployed as a PrometheusRule CRD to the Kubernetes cluster

apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: web-service-alerts
  namespace: monitoring
  labels:
    prometheus: kube-prometheus
    role: alert-rules
spec:
  groups:
    - name: web-service.latency
      interval: 30s
      rules:
        - alert: WebServiceHighP99Latency
          expr: |
            histogram_quantile(0.99,
              sum(rate(http_request_duration_seconds_bucket{service="web-service"}[5m]))
              by (le, service)
            ) > 2.0
          for: 3m
          labels:
            severity: warning
            team: platform
          annotations:
            summary: "p99 latency above 2s for web-service"
            description: "p99 latency is {{ $value | humanizeDuration }} for {{ $labels.service }}"
            runbook_url: "https://wiki.example.com/runbooks/web-service-latency"

        - alert: WebServiceHighErrorRate
          expr: |
            sum(rate(http_requests_total{service="web-service", status=~"5.."}[5m]))
            /
            sum(rate(http_requests_total{service="web-service"}[5m]))
            > 0.05
          for: 2m
          labels:
            severity: critical
            team: platform
          annotations:
            summary: "Error rate above 5% for web-service"
            description: "Error rate is {{ $value | humanizePercentage }}"
```

This PrometheusRule YAML is deployed the same way as every other Kubernetes manifest — via a GitOps pipeline. When an engineer wants to add a new alert or adjust a threshold, they open a PR. The PR shows the diff of the alert rule. A reviewer can evaluate whether the threshold is sensible. The change is deployed automatically on merge. No one needs to click through the Prometheus or Grafana UI to configure alerts, and no alert configuration lives only in a database that gets wiped on the next cluster migration.

The business case for monitoring-as-code is made most clearly by the alternative: a team migrates Grafana to a new server. The backup was supposed to run nightly but had been failing silently for three weeks due to a permissions issue. All dashboards are gone. The team spends two weeks recreating from memory, and the recreated dashboards miss several critical panels that only the engineer who wrote them remembered. This happens. It is entirely preventable.

---

## 6. What drift is and why it accumulates faster than you expect

Drift is the gap between three things that should be identical but usually are not: the **desired state** (what your Terraform code says should exist), the **Terraform state** (what Terraform's `terraform.tfstate` file believes exists), and the **actual state** (what the cloud provider actually has). When all three agree, you have no drift. When any of them disagree, you have drift.

The three-way split matters because the disagreement can happen at any of three distinct junctions, each requiring a different diagnosis:

1. **Code vs state drift**: you changed the Terraform code but haven't applied it yet. This is intentional and expected — it is the preview window before a change.

2. **State vs reality drift** (the dangerous kind): someone changed the cloud resource directly, without going through Terraform. Terraform's state still reflects the old reality. The code and state agree, but neither matches what actually exists. This is what `terraform plan` will show as unexpected changes on the next routine run.

3. **Code vs reality drift with a stale state**: the rarest but most dangerous kind. The state file was manually edited, or a resource was imported improperly, and now all three disagree with each other.

![Drift detection and reconciliation flow showing how scheduled CI branches into clean log or alert and three resolution paths](/imgs/blogs/iac-for-the-whole-stack-and-config-drift-3.png)

Drift accumulates because the forces creating it are constant and the forces detecting it are episodic. Every sprint someone uses the console to debug something. Every incident someone hotfixes a security group rule. Every quarter the security team adds an IAM permission for an audit. These are all small, well-intentioned changes. But without active detection, they pile up. By the time a team runs `terraform plan` after three months of quiet, the output can show dozens of unexpected changes, and no one can confidently say which ones are safe to apply.

### Why IAM is the #1 drift source

IAM is the most frequent drift source for three structural reasons:

First, IAM changes feel safe because they are not changing application code or infrastructure topology. Adding a policy to a role feels reversible and low-risk. So the bar to "just do it in the console" is lower than for, say, modifying a load balancer rule.

Second, IAM changes are often time-sensitive. A service is broken in production because it lacks a permission. The engineer opens the IAM console, adds the policy, the service works. The plan to "add it to Terraform" competes with the next sprint's tickets and usually loses.

Third, IAM resources have complex interdependencies. A role might be attached to a policy that is also attached to three other roles. The engineer who added the console permission does not know whether Terraform will recreate or modify it — so they are afraid to test the change in a plan. That fear of testing is itself a symptom of the coverage gap: in a fully IaC-managed environment, you would run `terraform plan` against a staging environment first, confirm the change is safe, then apply to production. The two-step apply is only safe when staging matches production in its IaC coverage. When staging is partial and production is partial (but differently partial), `plan` output between environments is not comparable.

The consequence is that IAM drift is the most common cause of the "Terraform deleted something in production" incident. The resource was not in Terraform's state. Terraform's plan correctly identified that the resource existed outside its management. The engineer reviewing the plan saw "Remove policy: AllowSecretsManagerReadPaymentService" and either did not recognize its significance or did not trace what service depended on it. Apply ran. Service broke.

---

## 7. Detecting drift: the scheduled plan in CI

The most powerful tool against drift accumulation is a scheduled Terraform `plan` that runs against the production environment even when no one is deploying anything. The plan reads the actual cloud state, compares it against the Terraform code, and reports any differences. If the output is non-empty when no deployment was intended, that is a drift alert.

Here is a GitHub Actions workflow that runs a drift detection plan every six hours:

```yaml
# .github/workflows/drift-detection.yml

name: Terraform drift detection

on:
  schedule:
    # Run every 6 hours
    - cron: "0 */6 * * *"
  workflow_dispatch:  # Allow manual trigger

permissions:
  id-token: write
  contents: read
  issues: write

jobs:
  detect-drift:
    name: Detect infrastructure drift
    runs-on: ubuntu-latest
    env:
      TF_VAR_environment: production

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure AWS credentials via OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::${{ secrets.AWS_ACCOUNT_ID }}:role/terraform-drift-detector
          aws-region: us-east-1

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.8.0"
          terraform_wrapper: false

      - name: Terraform init
        run: terraform init -input=false
        working-directory: ./infra/production

      - name: Terraform plan — detect drift
        id: plan
        run: |
          terraform plan \
            -detailed-exitcode \
            -out=drift.tfplan \
            -input=false \
            -no-color 2>&1 | tee plan_output.txt
          echo "exit_code=${PIPESTATUS[0]}" >> "$GITHUB_OUTPUT"
        working-directory: ./infra/production

      - name: Parse drift result
        id: drift
        run: |
          EXIT_CODE="${{ steps.plan.outputs.exit_code }}"
          if [ "$EXIT_CODE" = "0" ]; then
            echo "has_drift=false" >> "$GITHUB_OUTPUT"
            echo "Drift check: CLEAN — no differences found"
          elif [ "$EXIT_CODE" = "2" ]; then
            echo "has_drift=true" >> "$GITHUB_OUTPUT"
            echo "Drift check: DRIFT DETECTED"
            grep -E "^\s+(~|\+|-)" plan_output.txt | head -50 || true
          else
            echo "has_drift=error" >> "$GITHUB_OUTPUT"
            echo "Drift check: PLAN ERROR"
          fi

      - name: Create GitHub issue for drift
        if: steps.drift.outputs.has_drift == 'true'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const planOutput = fs.readFileSync('infra/production/plan_output.txt', 'utf8');
            const summary = planOutput
              .split('\n')
              .filter(l => l.match(/^\s+(~|\+|-)/) || l.match(/Plan:/))
              .slice(0, 50)
              .join('\n');

            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Infrastructure drift detected — ${new Date().toISOString().split('T')[0]}`,
              body: `## Drift detected in production\n\n` +
                    `Terraform plan found differences between state and reality.\n\n` +
                    `**Run details**: ${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}\n\n` +
                    `**Summary**:\n\`\`\`\n${summary}\n\`\`\`\n\n` +
                    `**Remediation options**:\n` +
                    `1. Update the Terraform code to match reality (if the change was intentional)\n` +
                    `2. Run \`terraform apply\` to remove the drift (if the change was unintended)\n` +
                    `3. Run \`terraform import\` to adopt the resource (if it should be IaC-managed)\n\n` +
                    `**SLA**: Drift must be reconciled within 5 business days.`,
              labels: ['infrastructure-drift', 'platform-team'],
            });

      - name: Post Slack notification
        if: steps.drift.outputs.has_drift == 'true'
        uses: slackapi/slack-github-action@v1
        with:
          channel-id: ${{ secrets.SLACK_PLATFORM_CHANNEL }}
          payload: |
            {
              "text": ":warning: Infrastructure drift detected in production",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": ":warning: *Infrastructure drift detected* — ${{ github.repository }}\n<${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View plan output>"
                  }
                }
              ]
            }
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
```

The key here is the `terraform plan -detailed-exitcode` flag. Exit code 0 means no differences. Exit code 1 means an error. Exit code 2 means the plan succeeded and found differences — this is the drift signal. By checking exit codes rather than parsing plan output, the detection is reliable even when output format changes across Terraform versions.

A subtlety: the drift detection role needs the same read permissions as the deploy role, but must have no write permissions. Many teams make the mistake of using the same role for drift detection and deployment, which means the drift detection job could accidentally apply if someone changes the workflow. The separation is important: detection is a read operation, remediation is a write operation, and the two should use different IAM roles.

The Terraform plan for drift detection also serves a second function: it exercises the Terraform configuration against the live cloud APIs. If the plan fails due to a provider API error, a resource that was deleted outside Terraform, or a configuration syntax error that was introduced since the last apply, the drift detection job will surface it. This means the drift detection CI is also a configuration health check — it tells you whether `terraform apply` would succeed before you try to run it during a deployment.

### Integrating drift detection with cost alerting

A valuable extension of the drift detection workflow is cost-impact reporting. When a drift item is found, Infracost can estimate the monthly cost delta:

```bash
# Extended drift detection step with cost estimation

terraform plan -out=drift.tfplan -input=false -no-color > plan_output.txt 2>&1
PLAN_EXIT=$?

if [ "$PLAN_EXIT" = "2" ]; then
  # Generate cost estimate for the drift
  infracost diff \
    --path drift.tfplan \
    --format json \
    --out-file infracost_output.json

  COST_DIFF=$(jq -r '.diffTotalMonthlyCost' infracost_output.json)
  echo "Estimated monthly cost impact of drift: \$$COST_DIFF"
fi
```

This matters because not all drift is operationally concerning — some drift represents cost creep. A developer who upgraded an RDS instance class via the console to handle a load test and never reverted it creates drift that adds hundreds of dollars per month to the bill. Without cost-impact reporting on drift, this kind of change is invisible until the monthly AWS invoice arrives.

The `terraform-drift-detector` IAM role used for this workflow is a read-only role. It can call `terraform plan` (which only reads cloud APIs) but cannot call `terraform apply`. This enforces the principle that drift detection does not accidentally fix what it finds — remediation is a deliberate human action.

![Coverage matrix showing which stack layers are commonly IaC managed, which are common drift sources, and the consequence of drift](/imgs/blogs/iac-for-the-whole-stack-and-config-drift-4.png)

---

## 8. The drift accumulation timeline: how bad it gets

The typical trajectory from zero to painful is fast. The first two weeks after an environment is provisioned with Terraform, it stays clean — no one has had time to make manual changes yet. Week three, an engineer debugging a connection issue opens a security group and adds a rule. Week four, a developer needs to test a new Lambda function and creates an IAM role with full access to S3 because it is faster than figuring out the minimum permissions. Month two, someone changes a CloudFront cache TTL via the console. Month three, a database DBA modifies a parameter group setting for performance.

By month four, `terraform plan` on the production environment shows 14 changes. Most of them are benign. One is the IAM role. But the engineer running the plan does not know which one is the IAM role, and they do not have time to trace each of the 14 changes back to its origin.

![Timeline showing drift accumulation from clean birth through manual changes to incident and recovery](/imgs/blogs/iac-for-the-whole-stack-and-config-drift-5.png)

The numbers from teams that have tracked this are sobering. A team of eight engineers makes approximately 2–3 manual console changes per week that are not reflected in Terraform. Over 90 days, that is 80–120 untracked changes. A survey of practitioners I ran across three mid-size companies found that teams without drift detection tooling had an average of 23 unresolved drift items per environment at any given time, and at least one of those 23 would cause a problem on the next major Terraform apply.

The fix is not to trust engineers to be more disciplined. Engineers are not the problem — the incentive structure is. When adding a console change takes 30 seconds and writing it in Terraform takes 15 minutes (especially when you do not know Terraform well), every urgent situation will produce console changes. The fix is to make the console change structurally harder or structurally visible.

---

## 9. Drift reconciliation: the three options and when to use each

When `terraform plan` shows drift, you have three choices. The right choice depends on whether the drift was intentional and whether the drifted state should become the new baseline.

**Option A: Update the code to match reality.** This is the right choice when the manual change was intentional and correct — for example, a DBA changed a parameter group setting that improved performance. The change was good. The problem was that it bypassed the Terraform workflow. The fix is to update the Terraform code to reflect the current reality, then run `plan` again to confirm zero diff. This option normalizes the change and adds it to version control with a commit message explaining why.

**Option B: Re-apply Terraform to remove the drift.** This is the right choice when the manual change was unintentional, incorrect, or unauthorized. For example, a developer added an IAM permission to test something and forgot to remove it. The drift is a security risk. The fix is to review the plan carefully, confirm the deletion is safe, and apply. This is the riskiest option — you are deleting something from production — so it requires careful review and should be done with someone who understands the impact watching.

**Option C: `terraform import` to adopt the resource.** This is the right choice when a resource was created outside Terraform but is load-bearing and should remain. The resource needs to be adopted into Terraform's management so future changes go through the workflow. `terraform import` brings the resource into the state file without modifying it. After import, you write the corresponding Terraform code, run `plan` to confirm zero diff, and commit.

```bash
# Option C: adopting a manually-created IAM role via terraform import

# Step 1: identify the drift
terraform plan -out=drift.tfplan 2>&1 | grep -E "(must be replaced|will be created|will be destroyed)"

# Step 2: write the resource block in Terraform code
# (see the iam-ecs-role module above as a template)

# Step 3: import the existing resource
terraform import \
  module.iam_ecs_role.aws_iam_role.task \
  my-service-task-role

# Step 4: verify the import produced zero diff
terraform plan
# Expected: No changes. Your infrastructure matches the configuration.

# Step 5: commit the import with context
git add infra/
git commit -m "feat(iam): adopt my-service-task-role into Terraform

Role was created manually during incident response on 2026-05-15.
Now managed as code to prevent drift on future applies.

Ref: INC-2891"
```

The commit message for an import is worth getting right. It should explain why the resource existed outside Terraform, what incident or decision created it, and the ticket reference. This creates a historical record that makes the next audit much easier.

### The "drift as a PR" workflow

A cleaner version of reconciliation is to treat every detected drift as a pull request, not a manual operation. The workflow is:

1. The scheduled CI job detects drift and creates a GitHub issue with the plan output.
2. A platform engineer reviews the issue and determines which option (A, B, or C) is appropriate.
3. They create a branch, make the corresponding change (update code, or no code change if Option B), and open a PR with a reference to the drift issue.
4. The PR runs a `terraform plan` in CI, confirming the expected result.
5. The PR is reviewed, approved, and merged. The merge triggers a production apply.
6. The drift issue is closed.

This workflow ensures that drift reconciliation is auditable, reviewed, and safe. It also creates a historical record of every drift event and its resolution, which is invaluable for postmortems and security audits.

---

## 10. Config drift vs infrastructure drift: two different problems

Drift has two distinct flavors that require different solutions, and conflating them leads to fighting the wrong battle.

**Infrastructure drift** is what we have been discussing: a resource in the cloud has been changed outside the IaC workflow. A security group rule was added via the console. An IAM policy was attached manually. This is IaC's problem — the fix is better coverage and detection.

**Configuration drift** is different: two or more servers that were born identical have diverged over time through in-place modifications. Server A has had the security team's monitoring agent installed. Server B got a hotfix for a library vulnerability applied directly via SSH. Server C has a different version of the application config because an engineer edited it in place three months ago. None of these are in Terraform or in Git. They are in the filesystem state of the servers themselves.

![Before and after comparison of mutable servers diverging over time versus immutable fleet staying identical from the same golden image](/imgs/blogs/iac-for-the-whole-stack-and-config-drift-8.png)

Configuration drift is **immutable infrastructure's problem**, not IaC's per se. The canonical solution is to never modify running servers — instead, build a new machine image (AMI via Packer, Docker image via Dockerfile) that includes the change, and replace all running servers with the new image. This is the immutable infra pattern.

With immutable infra, the fleet starts from an identical golden image every time a server is launched. After six months of operation, every server is still identical because no in-place changes are allowed. The monitoring agent is baked into the AMI. Library patches are AMI rebuilds. Config changes are AMI rebuilds. The Terraform code references the AMI ID; when the AMI changes, a rolling deployment replaces the fleet.

The relationship between the two drift types:

| Type | Cause | Affects | Fix |
|------|-------|---------|-----|
| Infrastructure drift | Console/API changes outside IaC | Cloud resource state | IaC coverage + drift detection |
| Configuration drift | In-place SSH/shell changes on servers | OS/app state inside servers | Immutable infrastructure + image builds |

Both types share the same root: a manual change bypassed the version-controlled workflow. Both types have the same detection strategy: compare actual state against known-good baseline. And both types have the same organizational fix: make the in-band path (PR → CI → apply/deploy) faster and easier than the out-of-band path (console click, SSH command).

---

## 11. The whole-stack IaC maturity model

Most teams are somewhere between Level 1 and Level 2. Most production incidents caused by Terraform-related drift require Level 3 or higher to prevent.

![IaC maturity model tree showing the progression from Level 1-2 basics through Level 3-4 advanced capabilities](/imgs/blogs/iac-for-the-whole-stack-and-config-drift-7.png)

**Level 0 — No IaC.** Everything is clicked in the console. This is fine for proof-of-concept work. It is not fine for anything that runs in production longer than two weeks.

**Level 1 — Compute only.** EC2 instances, autoscaling groups, Lambda functions, and ECS services are in Terraform. VPC, subnets, and security groups might be. IAM, DNS, databases, CDN, and monitoring are all console-managed. This is where most teams start, and where many teams stay longer than they should.

**Level 2 — Networking + compute.** VPCs, subnets, route tables, security groups, load balancers, and compute are all in Terraform. IAM, DNS, and databases are still console-managed. This level eliminates the "wrong port open" drift incident but not the IAM or DNS drift incidents.

**Level 3 — IAM + DNS + managed services.** The critical jump. IAM roles, policies, and SCPs are in Terraform. Route 53 zones and records are in Terraform. RDS instances, parameter groups, and backup windows are in Terraform. ACM certificates are in Terraform. This is the level that eliminates the most common production drift incident.

**Level 4 — Monitoring + policy as code.** Grafana dashboards, CloudWatch alarms, Prometheus alert rules, PagerDuty escalation policies, and OPA/Sentinel policies are all in Terraform or in version-controlled config files. Drift detection CI is running on a schedule. The no-console-in-prod SCP is enforced at the organization level. This is full whole-stack IaC coverage.

What distinguishes Level 4 from Level 3 is not just the additional resource types — it is the organizational posture. At Level 3, engineers can still make console changes; they just leave drift behind. At Level 4, console writes are structurally blocked. Every change requires a PR. Every PR runs a plan. Every plan output is reviewed. This is the same standard we apply to application code (no direct pushes to main, every change reviewed) applied to infrastructure.

The progression from Level 0 to Level 4 is a multi-year journey for most teams. The right order is to move one level at a time, fully, rather than trying to reach Level 4 in six months by partially covering everything. Partial coverage at any level is worse than full coverage at a lower level — because partial coverage creates the false confidence that Terraform is authoritative, which leads to unreviewed applies.

The table below shows what you gain at each transition:

| From → To | What gets codified | Drift incidents eliminated |
|-----------|-------------------|--------------------------|
| L0 → L1 | Compute, autoscaling | Compute config drift, instance-type drift |
| L1 → L2 | Networking, LB, SGs | Port-open drift, routing drift |
| L2 → L3 | IAM, DNS, RDS | IAM deletion incidents, DNS disappearance, DB param drift |
| L3 → L4 | Monitoring, policy | Silent alert outages, dashboard loss, policy bypass |

The most high-leverage transition is L2 → L3. Getting IAM, DNS, and managed databases into Terraform eliminates the most common and most painful drift incidents.

### Measuring your current maturity level

If you are not sure where your team sits on this model, the fastest way to find out is to run `terraform plan` against your production environment right now, without making any code changes, and count the unexpected changes in the output. Zero means you are at or near full coverage for the resources Terraform manages. Ten to twenty means you are mid-spiral. More than thirty means you have a significant coverage gap and should prioritize IaC expansion before the next major apply.

A second signal is the set of AWS services that appear in your CloudTrail logs but not in your Terraform state. CloudTrail records every API call in your AWS account. By comparing the set of resource ARNs in your Terraform state against the set of resource ARNs that appear in CloudTrail `CreateResource` events over the last 90 days, you can identify every resource that was created outside Terraform. This is the foundation of a drift audit.

```bash
# Quick drift inventory: count unmanaged IAM roles
# Compare IAM roles in AWS against roles in Terraform state

# List all IAM roles in AWS
aws iam list-roles --query "Roles[*].RoleName" --output text | tr '\t' '\n' | sort > /tmp/aws_roles.txt

# List all IAM roles managed by Terraform
terraform state list | grep aws_iam_role | sed 's/.*\.//' | sort > /tmp/tf_roles.txt

# Find roles in AWS but not in Terraform
comm -23 /tmp/aws_roles.txt /tmp/tf_roles.txt > /tmp/unmanaged_roles.txt
echo "Unmanaged IAM roles ($(wc -l < /tmp/unmanaged_roles.txt)):"
cat /tmp/unmanaged_roles.txt
```

Running this script on a mid-size AWS account typically reveals dozens of IAM roles created by developers over the years — testing roles, temporary roles, roles for AWS services that were configured manually. Each one is a potential drift bomb. The output becomes the prioritized backlog for the L2 → L3 IaC migration.

---

## 12. War story: the IAM deletion that caused a 47-minute outage

This is a real incident pattern, reproduced here with details changed to be representative rather than identifying. The team had good Terraform discipline — compute, networking, and most managed services were well-covered. IAM was "mostly in Terraform." The problem with "mostly" is that Terraform does not understand "mostly."

The sequence was:

1. A developer was debugging a permission error for the payment processor Lambda function. It needed to call Secrets Manager to retrieve a database password. The correct fix was to add the permission to the existing Terraform-managed role. The developer did not have time to trace the Terraform code, so they opened the IAM console and attached the `SecretsManagerReadAccess` policy directly to the role. The Lambda function worked immediately. The developer made a note in the ticket to "clean up the IAM in Terraform" and moved on.

2. The note sat in the ticket backlog for four months.

3. Four months later, the platform team ran a Terraform apply to update the Lambda function's memory and timeout settings. The plan showed three changes: two modifications to Lambda function configurations, and one change labeled "Remove policy attachment: arn:aws:iam::aws:policy/SecretsManagerReadAccess from role payment-processor-role."

4. The engineer reviewing the plan glanced at the third item, saw it was a policy removal from a Lambda role, and assumed it was cleaning up an over-privileged role from a previous sprint. They applied.

5. The payment processor Lambda started returning errors immediately. Checkout was broken. On-call paged. The incident response was 47 minutes long — 12 minutes to identify the cause (checking Lambda CloudWatch logs), 5 minutes to reattach the policy via the console, 30 minutes to write and deploy a hotfix that added the permission to Terraform properly.

The postmortem was run the next morning, with eleven people in the call. This is worth noting: a 47-minute outage with a team of eight engineers costs approximately \$3,000–\$5,000 in engineer-hours for the incident response and postmortem alone, before accounting for business impact (lost transactions, SLA credits, customer trust). The Terraform module to manage that IAM role properly would have taken one engineer about two hours to write. The ROI on IaC coverage is not abstract — it is measured in incidents averted.

The postmortem identified two contributing factors: (a) the original developer did not add the console change to Terraform, and (b) the engineer reviewing the plan did not recognize the significance of the policy removal. The corrective action was not "engineers must be more careful." It was: (a) drift detection CI would have surfaced the console change within six hours and created an issue; (b) a policy in the PR template now requires explicitly naming any IAM resource changes and their service impact.

A second war story worth noting: **the DNS record that disappeared after a Terraform run.** A team was managing Route 53 via Terraform but had a `_dmarc` TXT record that was added manually for email authentication purposes. During a migration of the hosted zone to a new AWS account, the Terraform code was copied but the `_dmarc` record was not in the code. The apply created the new hosted zone without it. Email for the domain stopped being deliverable for two days until the postmortem traced the issue to the missing DMARC record. The cost was not an outage — it was undelivered transactional email (order confirmations, password resets) and reputational damage with email providers.

---

## 13. The no-console-in-prod policy: the organizational fix

Technical drift detection catches drift after it happens. The organizational fix is to prevent drift from being created in the first place. The most effective single policy is **no write access to the production AWS console for anyone except the pipeline role and a break-glass emergency role**.

![Before and after comparison showing console access for all causing drift versus read-only console with IaC-only changes producing clean plans](/imgs/blogs/iac-for-the-whole-stack-and-config-drift-6.png)

Implementing this requires three things:

1. **An SCP** (like the one shown in Section 3) that denies all write API calls in production, with exceptions for the Terraform pipeline role and a break-glass role.

2. **A break-glass procedure** for genuine emergencies. When production is down and the fix requires a console change, the on-call engineer can assume the break-glass role, make the change, and the action is logged in CloudTrail. The post-incident requirement is to add the change to Terraform within 24 hours.

3. **Developer sandbox accounts** where engineers have full access for experimentation. The "I need to test this quickly" need is real and valid — it should be met by a sandbox environment, not by allowing console writes in production.

The cultural shift this requires is real. Developers who are used to having production console access will resist losing it. The most effective argument is concrete: show them the incident postmortems where console access caused outages. The second most effective argument is that with proper IaC coverage, their legitimate needs (adding a permission, changing a parameter) are met faster through the PR workflow than through a console change that creates drift and anxiety.

The PR workflow, once optimized, takes about 15 minutes end-to-end for a simple change: write the Terraform change (5 minutes if you have a good module), open a PR, the CI runs `plan` automatically (3 minutes), a colleague reviews the plan output (5 minutes), merge, and the pipeline applies. That is faster than the alternative for anyone who has experienced a post-apply incident: 30 seconds to make the console change, 47 minutes to debug the outage it eventually causes.

### The break-glass exception

The no-console-in-prod policy must have a clearly defined exception for genuine emergencies. Without it, you will face a political battle: engineers will resist the policy because they fear being unable to respond to incidents.

The break-glass exception works as follows. The break-glass role exists in IAM with full write access. Assuming it requires MFA and is limited to a specific set of trusted individuals (typically on-call leads). Every assumption of the break-glass role triggers an immediate CloudTrail alarm and a Slack notification to the platform team. The on-call engineer assumes the role, makes the minimum change needed to restore service, and the policy requires that the change be codified in Terraform within 24 hours with a postmortem reference.

```hcl
# modules/break-glass/main.tf

resource "aws_iam_role" "break_glass" {
  name = "break-glass-emergency"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        AWS = var.trusted_principal_arns
      }
      Condition = {
        Bool = { "aws:MultiFactorAuthPresent" = "true" }
        NumericLessThan = { "aws:MultiFactorAuthAge" = "300" }  # MFA within 5 minutes
      }
    }]
  })

  tags = {
    Purpose   = "Emergency break-glass access"
    ManagedBy = "terraform"
  }
}

# CloudWatch alarm when break-glass role is assumed
resource "aws_cloudwatch_event_rule" "break_glass_assumed" {
  name        = "break-glass-role-assumed"
  description = "Alert when the break-glass emergency role is assumed"

  event_pattern = jsonencode({
    source      = ["aws.sts"]
    detail-type = ["AWS API Call via CloudTrail"]
    detail = {
      eventName    = ["AssumeRole"]
      requestParameters = {
        roleArn = [aws_iam_role.break_glass.arn]
      }
    }
  })
}
```

The CloudWatch event rule fires the moment the break-glass role is assumed, before any console changes are made. This gives the platform team awareness and a head start on understanding what emergency is happening. The post-incident requirement to codify the change closes the drift loop — the emergency action becomes the PR, reviewed and merged, closing the incident ticket.

A before-and-after measurement from a team that implemented the no-console policy: before, the team ran drift detection monthly and typically found 15–25 drift items per environment. After implementing the no-console SCP, they ran drift detection weekly for the first six months and found fewer than two items per week, and those were almost always legitimate emergency break-glass changes that had been properly documented.

---

#### Worked example: whole-stack Terraform for a production service

Here is a complete worked example wiring together the modules from the previous sections. This is the Terraform for a production-grade three-tier web service: CloudFront in front, an ALB, an ECS service, and an RDS database.

```hcl
# infra/production/main.tf

terraform {
  required_version = "~> 1.8"

  backend "s3" {
    bucket         = "my-company-terraform-state"
    key            = "production/web-service/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    grafana = {
      source  = "grafana/grafana"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# Alias for us-east-1 — required for ACM certificates used with CloudFront
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

locals {
  service_name = "web-service"
  environment  = "production"
  tags = {
    Service     = local.service_name
    Environment = local.environment
    ManagedBy   = "terraform"
    Team        = "platform"
  }
}

# --- DNS + TLS ---
module "dns_tls" {
  source = "../../modules/dns-tls"

  domain_name            = "example.com"
  cloudfront_domain_name = module.cloudfront.domain_name
  cloudfront_zone_id     = module.cloudfront.hosted_zone_id
  tags                   = local.tags

  providers = {
    aws           = aws
    aws.us_east_1 = aws.us_east_1
  }
}

# --- CDN ---
module "cloudfront" {
  source = "../../modules/cloudfront"

  domain_name        = "example.com"
  certificate_arn    = module.dns_tls.certificate_arn
  origin_domain_name = module.alb.dns_name
  tags               = local.tags
}

# --- IAM ---
module "ecs_task_role" {
  source = "../../modules/iam-ecs-role"

  service_name = local.service_name
  secrets_arns = [
    "arn:aws:secretsmanager:us-east-1:123456789:secret:production/web-service/db-password",
    "arn:aws:secretsmanager:us-east-1:123456789:secret:production/web-service/api-keys",
  ]
  tags = local.tags
}

# --- Database ---
module "rds" {
  source = "../../modules/rds"

  identifier            = "${local.service_name}-${local.environment}"
  engine_version        = "8.0.35"
  instance_class        = "db.t3.medium"
  allocated_storage     = 100
  db_name               = "webservice"
  username              = "admin"
  password              = var.db_password
  subnet_ids            = module.vpc.private_subnet_ids
  vpc_security_group_ids = [module.security_groups.rds_sg_id]
  backup_retention_days = 14
  tags                  = local.tags
}

# --- Monitoring ---
module "monitoring" {
  source = "../../modules/monitoring"

  service_name      = local.service_name
  alert_webhook_url = var.slack_alert_webhook_url
}
```

The result is a production environment where every layer — DNS, TLS, CDN, IAM, RDS, monitoring — is expressed in code that is version-controlled, reviewed via PR, and applied by the pipeline. A `terraform plan` on this environment should show zero diff on any day when no code changes have been made. That is the invariant you are maintaining.

Before and after measurements from teams that have reached this level of coverage:

| Metric | Before full IaC coverage | After full IaC coverage |
|--------|--------------------------|------------------------|
| Unexpected changes per apply | 4–8 on average | 0 (only planned changes) |
| Time to diagnose drift incident | 45–90 minutes | 6–12 minutes (drift CI has a head start) |
| IAM-related outages per year | 2–3 | 0 |
| Security audit prep time | 3–5 days | 4 hours |
| Change review confidence | Low — "what is this change?" | High — every change has a PR |
| Lead time for IAM change | 30 sec (console) but breaks prod | 15 min (PR) — no prod risk |

---

#### Worked example: the drift-reconciliation PR

Here is a concrete example of working through a drift detection finding using the "drift as a PR" workflow.

**Situation**: the scheduled drift detection job runs and finds this in the plan output:

```bash
# Simulated terraform plan output showing drift

  # module.ecs_task_role.aws_iam_role_policy_attachment.extra[0] will be destroyed
  # (because aws_iam_role_policy_attachment.extra is not in configuration)
  - resource "aws_iam_role_policy_attachment" "extra" {
      - id         = "web-service-task-role/arn:aws:iam::aws:policy/AmazonDynamoDBReadOnlyAccess"
      - policy_arn = "arn:aws:iam::aws:policy/AmazonDynamoDBReadOnlyAccess"
      - role       = "web-service-task-role"
    }

  # module.cloudfront.aws_cloudfront_distribution.main will be updated in-place
  ~ resource "aws_cloudfront_distribution" "main" {
      ~ default_cache_behavior {
          ~ default_ttl = 86400 -> 3600
        }
    }
```

**Analysis**:
- The DynamoDB policy attachment is drift — someone manually attached it. Check with the service team whether the web service actually uses DynamoDB. If not, it is over-privilege and should be removed (Option B). If yes, add it to the Terraform module (Option A).
- The CloudFront TTL change is drift — someone lowered the cache TTL via the console, probably to debug a caching issue. Check whether the lower TTL was intended as permanent. If yes, update the Terraform code (Option A). If not, the re-apply will restore the 24-hour TTL (Option B).

**Resolution for the DynamoDB attachment** (the service team confirms it is not needed):

```bash
# Option B — re-apply to remove unauthorized policy

# Step 1: review the plan one more time
terraform plan -target=module.ecs_task_role

# Step 2: confirm with the service team
# (check Slack, ticket history, or DynamoDB access logs)

# Step 3: apply the removal with a narrow target
terraform apply \
  -target=module.ecs_task_role.aws_iam_role_policy_attachment.extra \
  -auto-approve=false

# Output:
# module.ecs_task_role.aws_iam_role_policy_attachment.extra[0]: Destroying...
# Apply complete! Resources: 0 added, 0 changed, 1 destroyed.
```

**Resolution for the CloudFront TTL change** (the team wants to keep the lower TTL):

```hcl
# Option A — update Terraform code to match reality
# In modules/cloudfront/main.tf, change:
#   default_ttl = 86400
# to:
#   default_ttl = 3600

# Commit with context
# "fix(cloudfront): reduce default TTL to 3600s
#
# Console change from 2026-06-20 debugging session (lower cache TTL
# improved A/B test feedback latency). Keeping the change as intentional.
#
# Ref: drift-detection run 2026-06-22"
```

Running `terraform plan` after both changes should show zero diff. The PR closes the drift issue. The whole sequence takes about 30 minutes — the same time it used to take to just identify the cause of a drift incident.

---

## 14. When whole-stack IaC is not worth it — and when it absolutely is

Whole-stack IaC coverage has real costs. The Terraform module for a CloudFront distribution is about 80 lines of HCL. The Grafana dashboard-as-code workflow requires engineers to export JSON and commit it. The SCP for IAM requires organizational AWS accounts. None of this is free.

**When it is probably not worth it**:

- A three-person startup with a single production environment. Your time is better spent building product. Use a PaaS (Render, Railway, Fly.io) that manages this layer for you.
- A prototype or internal tool with fewer than 10 users. The cost of the drift management workflow exceeds the cost of the occasional drift incident.
- When you do not have a dedicated platform or infrastructure team. Without ownership, IaC for the whole stack becomes a liability — a partially-maintained Terraform repo that no one really understands.

**When it absolutely is worth it**:

- Any system where an IAM change can cause a production outage. That is any system that uses IAM-controlled AWS services — which is almost everything on AWS.
- Any system subject to compliance requirements (SOC 2, PCI DSS, HIPAA). Auditors want to see infrastructure changes reviewed and approved. A Git history with PR reviews is the cleanest audit trail.
- Any team where more than two people make infrastructure changes. Console-only workflows do not scale — knowledge is siloed, changes conflict, and tribal memory becomes a dependency.
- Any system with uptime requirements above 99.5%. At that level, a 47-minute IAM-deletion outage is a serious SLA breach.

The broader principle: treat IaC coverage as a risk management tool, not a technical ideal. The question is not "should everything be in Terraform?" but "what is the cost of this resource drifting, and does that exceed the cost of codifying and managing it?" For IAM and DNS, the answer is almost always yes — the blast radius of drift is enormous. For a Grafana dashboard for a dev environment, the answer is probably no.

---

## 15. Key takeaways

- **IaC coverage is binary per resource**: a resource is either in Terraform state or it is not. "Mostly managed" is the same as "not managed" when `apply` runs. Partial coverage creates drift traps.

- **IAM is the #1 drift source and the highest blast-radius**: manual IAM changes are common, feel low-risk, and are the most likely to cause a production outage when Terraform removes them on the next apply. Get IAM fully into Terraform before any other resource type.

- **DNS is the second-highest blast-radius drift source**: a missing DNS record makes your service unreachable globally. A missing DMARC record breaks email deliverability. Both happen silently on Terraform applies when records are not in code.

- **Drift accumulates faster than teams expect**: even disciplined teams make 2–3 console changes per week. Without detection, that is 100+ untracked changes per environment per year.

- **The scheduled drift detection CI job is the most important automation you are not running**: a `terraform plan -detailed-exitcode` every six hours catches drift before it causes incidents. It is a 50-line GitHub Actions workflow. There is no excuse not to have it.

- **The three reconciliation options have distinct use cases**: update-code for intentional changes, re-apply for unauthorized changes, `terraform import` for load-bearing orphaned resources. Mixing them up is how you accidentally delete production resources.

- **Config drift and infrastructure drift require different fixes**: infrastructure drift is an IaC coverage and detection problem; config drift is an immutable infrastructure problem. Terraform cannot fix servers that have diverged via SSH.

- **The no-console-in-prod SCP is the organizational fix**: drift detection catches drift after it happens; the SCP prevents it from being created. Both are required for a mature posture.

- **Whole-stack IaC maturity has four levels**: L1 (compute) → L2 (networking) → L3 (IAM+DNS+managed services) → L4 (monitoring+policy). The L2→L3 transition eliminates the majority of production drift incidents.

- **Drift reconciliation should be a PR, not a manual operation**: creating a GitHub issue for every detected drift event, resolving it via a PR with a `terraform plan` in CI, and closing the issue on merge creates an audit trail, forces review, and prevents accidental production changes.

---

## Further reading

- **[From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model)** — the full spine this post fits into; the "everything as code" principle.
- **[Infrastructure as code: from ClickOps to declarative](/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative)** — the foundations of IaC; start here if Terraform is new to you.
- **[Managing Terraform safely at scale](/blog/software-development/ci-cd/managing-terraform-safely-at-scale)** — remote state, workspaces, Atlantis/Terraform Cloud, and the safe-apply workflow.
- **GitOps: Git as the source of truth** — the pull-based GitOps model extends the IaC "everything in Git" principle into application deployments via Argo CD and Flux, covered in the next track of this series.
- **[Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure)** — the SRE side of drift: what happens when your infrastructure does not match your expectations under load.
- **[Terraform documentation: Import](https://developer.hashicorp.com/terraform/language/import)** — the official reference for `terraform import` and the newer `import` block syntax.
- **[AWS Organizations SCPs documentation](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_manage_policies_scps.html)** — how to implement the no-console-in-prod policy at the organization level.
- **[Driftctl](https://driftctl.com/)** / **[Infracost](https://www.infracost.io/)** — open-source tools for drift detection and cost tracking in Terraform workflows.
