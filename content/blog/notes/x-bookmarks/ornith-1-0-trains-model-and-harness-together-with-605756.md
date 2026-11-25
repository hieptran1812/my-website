---
title: "Ornith-1.0 Trains Model And Harness Together With RL"
publishDate: "2026-06-25"
date: "2026-06-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "reinforcement-learning", "harness-engineering"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2070155810995605756-1.jpg"
excerpt: "A post says Ornith-1.0 is unique because RL jointly trains the model with its testing harness instead of using a fixed one."
---

> Source: [@ai_for_success on X](https://x.com/ai_for_success/status/2070155810995605756) · posted 2026-06-25

![Ornith-1.0 Trains Model And Harness Together With RL (1)](/imgs/blogs/x-bookmarks/2070155810995605756-1.jpg)

![Ornith-1.0 Trains Model And Harness Together With RL (2)](/imgs/blogs/x-bookmarks/2070155810995605756-2.jpg)

## Summary

@ai_for_success says open-source models are catching up to closed ones for agentic coding, and Ornith-1.0 (9B dense up to 397B MoE) just dropped. The unique part is training: instead of a fixed testing harness, it uses reinforcement learning to jointly train, per the post. The text is cut off.

## Background

A harness is the scaffold around a model that gives it tools, prompts and tests. Normally it is fixed while the model is trained. Training both together could let the model and its scaffold adapt to each other.

## What I took away

- Model and harness can be co-trained.
- Harness design is part of agent performance, not just the model.

## Reference

- Original post: [https://x.com/ai_for_success/status/2070155810995605756](https://x.com/ai_for_success/status/2070155810995605756)
- Author: [@ai_for_success](https://x.com/ai_for_success) on X, posted 2026-06-25
