---
title: "Vibe RL: Teaching Qwen3-4B to Draw ASCII Diagrams"
publishDate: "2026-03-06"
date: "2026-03-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2029764051597766794-1.jpg"
excerpt: "A blog about using Prime Lab and Codex to RL-train Qwen3-4B to beat GPT 4.1 mini on ASCII diagrams."
---

> Source: [@27upon2 on X](https://x.com/27upon2/status/2029764051597766794) · posted 2026-03-06

![Vibe RL: Teaching Qwen3-4B to Draw ASCII Diagrams](/imgs/blogs/x-bookmarks/2029764051597766794-1.jpg)

## Summary

The author wrote a blog on Vibe RL and their experience using the Prime Lab and Codex to get Qwen3-4B to draw better ASCII diagrams than GPT 4.1 mini by about 2x, as they claim. Takeaways include using an agent-friendly RL stack like PrimeIntellect and vibe-coding an RL studio to spot reward hacking.

## Background

Reinforcement learning (RL) trains a model using reward signals. Reward hacking is when the model finds ways to score high without doing the real task. A tool to inspect outputs helps catch that.

## What I took away

- Inspect rollouts to detect reward hacking.
- Small models can beat larger ones on narrow tasks with RL.
- Agent-friendly RL tooling speeds up experiments.

## Reference

- Original post: [https://x.com/27upon2/status/2029764051597766794](https://x.com/27upon2/status/2029764051597766794)
- Author: [@27upon2](https://x.com/27upon2) on X, posted 2026-03-06
