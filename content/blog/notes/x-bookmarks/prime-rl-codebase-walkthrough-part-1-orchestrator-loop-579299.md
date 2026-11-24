---
title: "Prime-RL Codebase Walkthrough Part 1 Orchestrator Loop"
publishDate: "2026-08-03"
date: "2026-08-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Part 1 of a blog series that reads the prime-rl codebase, covering the Orchestrator's main loop."
---

> Source: [@vivek_2332 on X](https://x.com/vivek_2332/status/2084291916221579299) · posted 2026-08-03

## Summary

The author says they have been exploring RL infrastructure and want to go deep into it. They are reading the **prime-rl** codebase from Prime Intellect and turning it into a blog series that explains key concepts through code. Part 1 covers the Orchestrator's main loop.

## Background

prime-rl is Prime Intellect's framework for RL training of language models. The orchestrator coordinates the main stages: sending prompts for generation, scoring rollouts and passing batches to the trainer.

## What I took away

- The orchestrator loop is the best entry point to an RL training system.
- Writing a series while reading code is a strong way to learn.
- Part 2 in this batch covers the RolloutDispatcher.

## Reference

- Original post: [https://x.com/vivek_2332/status/2084291916221579299](https://x.com/vivek_2332/status/2084291916221579299)
- Author: [@vivek_2332](https://x.com/vivek_2332) on X, posted 2026-08-03
