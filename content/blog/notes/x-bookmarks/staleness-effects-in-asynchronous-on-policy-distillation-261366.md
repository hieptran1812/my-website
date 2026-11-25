---
title: "Staleness Effects In Asynchronous On-Policy Distillation"
publishDate: "2026-06-24"
date: "2026-06-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2069719515429261366-1.jpg"
excerpt: "A post studies how stale-policy data affects on-policy distillation when rollouts are made asynchronous."
---

> Source: [@sheriyuo on X](https://x.com/sheriyuo/status/2069719515429261366) · posted 2026-06-24

![Staleness Effects In Asynchronous On-Policy Distillation](/imgs/blogs/x-bookmarks/2069719515429261366-1.jpg)

## Summary

@sheriyuo says OPD has the same systems bottleneck as RL: rollouts dominate training time on reasoning workloads. Going async improves throughput but feeds the learner stale-policy data, and the effect on OPD was unstudied. The clean finding is about KL, but the text is cut off.

## Background

Rollouts are the sampled model outputs used for training. Asynchronous training overlaps generation and learning for speed, but the generator can lag behind the learner, producing stale data. KL measures how far two distributions differ.

## What I took away

- Async training buys throughput at the cost of stale data.
- OPD staleness had been unstudied until this work.

## Reference

- Original post: [https://x.com/sheriyuo/status/2069719515429261366](https://x.com/sheriyuo/status/2069719515429261366)
- Author: [@sheriyuo](https://x.com/sheriyuo) on X, posted 2026-06-24
