---
title: "Scaling RL on TPUs for the MiMo Models"
publishDate: "2026-09-21"
date: "2026-09-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "hardware", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "Full-parameter RL on 310B MiMo-V2.6 across 1,000+ TPUs, where scaling with JAX was mostly a config change."
---

> Source: [@tianjun_zhang on X](https://x.com/tianjun_zhang/status/2102152886520291472) · posted 2026-09-21

## Summary

The author describes bringing RL to scale with JAX and TPUs for the MiMo family, saying scaling up was mostly a config change, not a code rewrite. The quoted post reports full-parameter RL on MiMo-V2.6 at 310B and other stable runs of 1,000+ steps across 1,000+ TPUs. It also mentions optimized vLLM inference for faster rollouts and bitwise matching.

## Background

Full-parameter RL updates all model weights, unlike lighter methods that train small adapters. JAX expresses sharding declaratively, so changing scale often means changing configuration. Bitwise matching between the rollout engine and the trainer avoids subtle mismatches that destabilize RL.

## What I took away

- Frameworks where scale is a config change save huge engineering effort.
- Keep inference and training numerically aligned in RL pipelines.
- TPUs are a viable option for large-scale RL post-training.

## Reference

- Original post: [https://x.com/tianjun_zhang/status/2102152886520291472](https://x.com/tianjun_zhang/status/2102152886520291472)
- Author: [@tianjun_zhang](https://x.com/tianjun_zhang) on X, posted 2026-09-21
