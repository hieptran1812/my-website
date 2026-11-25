---
title: "Unsloth Long-Context RL Training"
publishDate: "2026-01-15"
date: "2026-01-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2011827592886960131-1.jpg"
excerpt: "Unsloth says GRPO training now supports 7x longer context, reaching 380K on a 192GB GPU for gpt-oss."
---

> Source: [@UnslothAI on X](https://x.com/UnslothAI/status/2011827592886960131) · posted 2026-01-15

![Unsloth Long-Context RL Training](/imgs/blogs/x-bookmarks/2011827592886960131-1.jpg)

## Summary

Unsloth announces reinforcement learning training with 7x longer context and no accuracy loss, using new batching algorithms. They say gpt-oss can be trained with GRPO to 380K context on a 192GB GPU. These are the vendor's claims.

## Background

GRPO is an RL method for LLMs that compares groups of sampled answers. Long reasoning chains make RL memory heavy, so batching improvements matter.

## What I took away

- Memory-efficient batching makes long-context RL more practical.
- Verify claims on your own hardware.

## Reference

- Original post: [https://x.com/UnslothAI/status/2011827592886960131](https://x.com/UnslothAI/status/2011827592886960131)
- Author: [@UnslothAI](https://x.com/UnslothAI) on X, posted 2026-01-15
