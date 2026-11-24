---
title: "llm-d Flow Control Solves the Noisy Neighbor Problem"
publishDate: "2026-07-30"
date: "2026-07-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Red Hat AI describes how llm-d flow control uses priority queuing and fairness to protect interactive users from batch jobs on a shared model."
---

> Source: [@RedHat_AI on X](https://x.com/RedHat_AI/status/2082835471940366657) · posted 2026-07-30

## Summary

Red Hat AI describes the **noisy-neighbor problem**: two teams share a model, one running batch summarization and one a live coding chatbot. The batch job saturates the GPU and the chatbot's time-to-first-token spikes. They say flow control in llm-d fixes this with priority-based queuing and fairness, and link a Red Hat Developer article on OpenShift AI.

## Background

Time-to-first-token (TTFT) is how long a user waits before the first word appears. When workloads share a GPU, heavy batch requests can crowd out latency-sensitive ones. Flow control decides which requests go first, so interactive traffic stays fast while batch work fills spare capacity.

## What I took away

- Separate traffic by priority when sharing models across teams.
- Batch and interactive workloads need different service levels.
- Track TTFT per workload class, not just overall.

## Reference

- Original post: [https://x.com/RedHat_AI/status/2082835471940366657](https://x.com/RedHat_AI/status/2082835471940366657)
- Author: [@RedHat_AI](https://x.com/RedHat_AI) on X, posted 2026-07-30
