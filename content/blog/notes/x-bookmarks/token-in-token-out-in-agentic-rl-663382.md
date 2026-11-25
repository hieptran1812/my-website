---
title: "Token-In-Token-Out in Agentic RL"
publishDate: "2026-06-09"
date: "2026-06-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-agents", "harness-engineering"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2064399810170663382-1.jpg"
excerpt: "A post on a blog explaining TITO, which makes sure the trainer sees the exact tokens the inference engine produced."
---

> Source: [@ying11231 on X](https://x.com/ying11231/status/2064399810170663382) · posted 2026-06-09

![Token-In-Token-Out in Agentic RL](/imgs/blogs/x-bookmarks/2064399810170663382-1.jpg)

## Summary

@ying11231 recalls a colleague pushing for TITO. The quoted blog says in agentic RL a rollout is a chain of model calls, tool outputs and resumed turns, and TITO ensures the trainer evaluates the exact tokens the engine produced.

## Background

Retokenizing text can give different tokens than were sampled, causing a mismatch between training and rollout. Using exact tokens avoids it.

## What I took away

- Keep token ids from rollout to training.
- Mismatches silently hurt RL.

## Reference

- Original post: [https://x.com/ying11231/status/2064399810170663382](https://x.com/ying11231/status/2064399810170663382)
- Author: [@ying11231](https://x.com/ying11231) on X, posted 2026-06-09
