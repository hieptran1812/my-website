---
title: "Sequence Level Versus Token Level Optimization in RL"
publishDate: "2026-07-26"
date: "2026-07-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "The author notes RL for LLMs aims at sequence-level goals but optimizes at the token level, an issue often ignored for KL losses."
---

> Source: [@HeMuyu0327 on X](https://x.com/HeMuyu0327/status/2081250230213939310) · posted 2026-07-26

## Summary

The author points out a mismatch in RL: we want to optimize at the **sequence level**, but in practice we optimize at the **token level**. For the policy gradient loss, GSPO addresses this well. For KL or OPD losses, the author says the issue is glossed over (text cut off).

## Background

In LLM RL, the reward is usually given for a whole answer (a sequence), but losses are computed per token. GSPO is a method that uses sequence-level importance ratios. KL loss keeps the model close to a reference, and OPD is on-policy distillation; both can have the same token versus sequence mismatch.

## What I took away

- Check whether your loss matches the level at which rewards are defined.
- Token-level approximations can introduce bias.
- KL and distillation losses deserve the same scrutiny as policy losses.

## Reference

- Original post: [https://x.com/HeMuyu0327/status/2081250230213939310](https://x.com/HeMuyu0327/status/2081250230213939310)
- Author: [@HeMuyu0327](https://x.com/HeMuyu0327) on X, posted 2026-07-26
