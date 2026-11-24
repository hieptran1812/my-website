---
title: "Actor-Critic With Action Chunking for LLM RL"
publishDate: "2026-10-02"
date: "2026-10-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "Tengyu Ma highlights AC2, which uses a learned critic to avoid finishing every rollout in LLM RL."
---

> Source: [@tengyuma on X](https://x.com/tengyuma/status/2106076689889652961) · posted 2026-10-02

## Summary

Tengyu Ma says he always thought GRPO-style RL could not be optimal and that efficient RL matters with billions going into self-play. He lists what he likes: critics supervising the policy, exploiting easy resets for LLMs, and sharing parameters between Q and policy. The quoted post introduces AC2, which scores chunks of tokens with a learned critic so only partial rollouts are needed, claimed to train faster than GRPO.

## Background

GRPO compares complete responses within a group to estimate how good each is, which requires finishing every rollout. A critic is a learned model that estimates future reward, so it can judge partial outputs.

## What I took away

- Partial rollouts could cut the cost of long-horizon RL.
- A critic must be accurate to be trusted; watch for bias.
- LLMs allow easy resets to any prefix, unlike robots.

## Reference

- Original post: [https://x.com/tengyuma/status/2106076689889652961](https://x.com/tengyuma/status/2106076689889652961)
- Author: [@tengyuma](https://x.com/tengyuma) on X, posted 2026-10-02
