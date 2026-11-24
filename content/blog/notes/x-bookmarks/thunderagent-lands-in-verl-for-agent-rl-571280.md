---
title: "ThunderAgent Lands in verl for Agent RL"
publishDate: "2026-07-29"
date: "2026-07-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "serving", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "A ThunderAgent author says the work now powers Together AI products and has landed in verl-recipe with a Dynamo recipe for agent RL."
---

> Source: [@Mlsys_HaoKang on X](https://x.com/Mlsys_HaoKang/status/2082616871279571280) · posted 2026-07-29

## Summary

The author is happy to see **ThunderAgent** help Together AI's agent RL and serving products. ThunderAgent has also landed in verl-recipe, where a new Dynamo recipe brings program-aware routing to multi-turn agent RL. They claim 1.94x rollout throughput and 1.39x full-step speedup (text cut off); the quoted post is Together AI's ThunderAgent announcement.

## Background

In RL training of agents, rollouts are the episodes where the model acts and collects rewards; they often dominate training time. verl is an open RL training framework for LLMs. Program-aware routing sends requests from the same agent program to the same server so cached context is reused.

## What I took away

- Rollout generation is often the bottleneck in agent RL.
- Inference serving tricks transfer directly to RL training speed.
- Keeping requests from the same session together improves cache reuse.

## Reference

- Original post: [https://x.com/Mlsys_HaoKang/status/2082616871279571280](https://x.com/Mlsys_HaoKang/status/2082616871279571280)
- Author: [@Mlsys_HaoKang](https://x.com/Mlsys_HaoKang) on X, posted 2026-07-29
