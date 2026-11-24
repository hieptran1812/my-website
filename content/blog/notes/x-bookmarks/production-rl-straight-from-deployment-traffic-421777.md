---
title: "Production RL Straight From Deployment Traffic"
publishDate: "2026-11-06"
date: "2026-11-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A recipe for improving a model with RL directly from its production data stream, praised as a top priority for companies."
---

> Source: [@samueldenton on X](https://x.com/samueldenton/status/2118790820132421777) · posted 2026-11-06

## Summary

The quoted post describes a recipe for production RL: a model steadily improves from its own deployment's data stream and is then redeployed. It argues production traffic is full of real, economically valuable experience that rarely makes it back into training. @samueldenton says that if he ran a company, getting this working would be his number one priority, because it skips RL environments, simulated tasks and large data-creation efforts.

## Background

Reinforcement learning (RL) trains a model by rewarding good outputs. Most labs build artificial environments or synthetic tasks for this. Learning directly from real user traffic means the training signal comes from the actual work the model is used for, which can be cheaper and more relevant, but needs care with feedback quality and privacy.

## What I took away

- Real usage data may be a more valuable training signal than synthetic RL environments.
- A closed loop of deploy, collect, train, redeploy is the core idea to understand.
- Ask how rewards are defined from production data and how regressions are caught before redeploying.

## Reference

- Original post: [https://x.com/samueldenton/status/2118790820132421777](https://x.com/samueldenton/status/2118790820132421777)
- Author: [@samueldenton](https://x.com/samueldenton) on X, posted 2026-11-06
