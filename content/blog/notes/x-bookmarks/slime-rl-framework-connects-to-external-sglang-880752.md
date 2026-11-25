---
title: "slime RL Framework Connects To External SGLang"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "slime lets an RL job use an already-running SGLang fleet for rollouts instead of owning its own servers."
---

> Source: [@slime_framework on X](https://x.com/slime_framework/status/2062788087168880752) · posted 2026-06-05

## Summary

The slime framework says RL frameworks are moving from engine mode to server mode. It goes further: the RL job does not need to own rollout servers. You can bring your own SGLang fleet managed by your serving system, and slime connects to and registers it.

## Background

In LLM RL, rollouts are the model's generated attempts that are scored and trained on. Running them on a shared serving fleet can improve hardware use and separates training from inference operations.

## What I took away

- Decoupling rollout servers from the trainer can reuse existing infrastructure.
- Server mode means the trainer talks to inference over an API.

## Reference

- Original post: [https://x.com/slime_framework/status/2062788087168880752](https://x.com/slime_framework/status/2062788087168880752)
- Author: [@slime_framework](https://x.com/slime_framework) on X, posted 2026-06-05
