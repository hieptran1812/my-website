---
title: "Prime-RL Codebase Walkthrough Part 2 RolloutDispatcher"
publishDate: "2026-08-07"
date: "2026-08-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Part 2 of a blog series reading the prime-rl codebase, focused on the RolloutDispatcher that sends prompts to inference and collects rollouts."
---

> Source: [@vivek_2332 on X](https://x.com/vivek_2332/status/2085738800953843756) · posted 2026-08-07

## Summary

The author releases part 2 of a series that explains the **prime-rl** codebase from Prime Intellect through its code. This part covers the RolloutDispatcher, which sends prompts to the inference servers and collects finished rollouts. The quoted post is part 1, where the author explains the series and walks through the Orchestrator's main loop.

## Background

In RL training for LLMs, a rollout is one generated answer (or episode) that is later scored with a reward. RL infrastructure must keep inference workers busy generating rollouts while the trainer updates weights, so the dispatch and collection logic is a core piece of the system.

## What I took away

- Reading a real RL training codebase is a good way to learn RL infrastructure.
- Rollout generation and training are separate stages that need careful coordination.
- Start from the main loop, then follow each component it calls.

## Reference

- Original post: [https://x.com/vivek_2332/status/2085738800953843756](https://x.com/vivek_2332/status/2085738800953843756)
- Author: [@vivek_2332](https://x.com/vivek_2332) on X, posted 2026-08-07
