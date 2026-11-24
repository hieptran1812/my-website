---
title: "Scaling RL Training to One Million Token Context"
publishDate: "2026-10-22"
date: "2026-10-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "A team shares how it supported 1M-token context training on SkyRL for GLM 5.3 Flash, far beyond a Megatron baseline."
---

> Source: [@robertnishihara on X](https://x.com/robertnishihara/status/2113314866983493688) · posted 2026-10-22

## Summary

@robertnishihara recommends a post from his team for the low-level algorithm and systems details of scaling training to 1M-token context. The quoted author says they supported 1M-token context training on SkyRL for GLM 5.3 Flash, 64x higher than the out-of-the-box Megatron baseline, in collaboration with Lila Sciences.

## Background

Context length is how many tokens a model sees at once. Training on very long sequences needs a lot of memory for activations and attention, so it requires techniques like splitting sequences across GPUs. SkyRL is a reinforcement learning training framework, and Megatron is a widely used large-model training library.

## What I took away

- Long-context training is mostly a memory and parallelism problem.
- Default framework settings can be far from what is possible.
- Read the post for concrete systems techniques.

## Reference

- Original post: [https://x.com/robertnishihara/status/2113314866983493688](https://x.com/robertnishihara/status/2113314866983493688)
- Author: [@robertnishihara](https://x.com/robertnishihara) on X, posted 2026-10-22
