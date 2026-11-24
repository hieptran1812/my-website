---
title: "Finetuning Through Sampling Rivals RL"
publishDate: "2026-10-02"
date: "2026-10-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
excerpt: "Researchers claim combining sampling with SFT matches RL in post-training while forgetting less."
---

> Source: [@du_yilun on X](https://x.com/du_yilun/status/2106044151892693270) · posted 2026-10-02

## Summary

The author says they found a way to make SFT match RL in post-training, called finetuning through sampling, which lets an LLM learn new knowledge without forgetting old information. The quoted post says SFT is not dead, claiming it often generalizes better and forgets less than RL and OPSD, building on earlier work on reasoning with sampling.

## Background

SFT (supervised fine-tuning) trains on example outputs directly, while RL trains on rewards from the model's own outputs. RL often forgets less because it learns from its own distribution; sampling-based SFT tries to get that benefit.

## What I took away

- Forgetting is a key metric when comparing post-training methods.
- SFT may still compete with RL when data comes from the model's own samples.
- These are the authors' claims; check the experiments.

## Reference

- Original post: [https://x.com/du_yilun/status/2106044151892693270](https://x.com/du_yilun/status/2106044151892693270)
- Author: [@du_yilun](https://x.com/du_yilun) on X, posted 2026-10-02
