---
title: "RL on Large MoEs With Zero Train-Inference Mismatch"
publishDate: "2026-08-17"
date: "2026-08-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "moe", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "A team reports RL training of large MoE models with no mismatch between training and inference engines."
---

> Source: [@PandaAshwinee on X](https://x.com/PandaAshwinee/status/2089396727048749528) · posted 2026-08-17

## Summary

The author says they can now run RL on large MoE models with zero train-inference mismatch, and that this can improve performance. Their example task teaches Qwen3.6-35B-A3B to play Wordle. Everything is open-source and the thread includes ablations.

## Background

In LLM RL, a fast inference engine generates samples while a separate training engine computes gradients. Small numerical differences between the two (a mismatch) mean the trainer scores tokens slightly differently than they were sampled, which can destabilize training. MoE models make this worse because tiny differences can change which experts are picked.

## What I took away

- Check for train-inference mismatch when RL runs behave strangely.
- MoE routing is sensitive to numerical noise, so consistency matters more there.
- Ablations in the thread are the place to see how much the fix helps.

## Reference

- Original post: [https://x.com/PandaAshwinee/status/2089396727048749528](https://x.com/PandaAshwinee/status/2089396727048749528)
- Author: [@PandaAshwinee](https://x.com/PandaAshwinee) on X, posted 2026-08-17
