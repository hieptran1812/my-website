---
title: "SGLang Rollout Routing Replay For MoE RL"
publishDate: "2025-12-23"
date: "2025-12-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "moe", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2003304473213698238-1.jpg"
excerpt: "An SGLang RL contributor celebrates Rollout Routing Replay support for stable RL training of MoE models, with the Qwen team."
---

> Source: [@GenAI_is_real on X](https://x.com/GenAI_is_real/status/2003304473213698238) · posted 2025-12-23

![SGLang Rollout Routing Replay For MoE RL](/imgs/blogs/x-bookmarks/2003304473213698238-1.jpg)

## Summary

The author thanks the Qwen team and the slime team for their work together. He says SGLang RL members have all moved on to new roles at several companies and invites others to join. The quoted post announces that SGLang and Miles now support Rollout Routing Replay (R3) for stable reinforcement learning on MoE models, which it says has been notoriously unstable.

## Background

A mixture-of-experts (MoE) model routes each token to a few expert sub-networks. In RL, the routing chosen during generation can differ from the routing during training, which can destabilize learning. Replaying the rollout's routing in training is meant to keep the two consistent.

## What I took away

- MoE routing mismatch between inference and training is a known source of RL instability.
- R3 replays the rollout routing during training to address it.
- SGLang and Miles now support this.

## Reference

- Original post: [https://x.com/GenAI_is_real/status/2003304473213698238](https://x.com/GenAI_is_real/status/2003304473213698238)
- Author: [@GenAI_is_real](https://x.com/GenAI_is_real) on X, posted 2025-12-23
