---
title: "Setting Batch Size for Large-Scale Pretraining"
publishDate: "2026-01-09"
date: "2026-01-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2009722476612272286-1.jpg"
excerpt: "InternLM research on how critical batch size behaves under Warmup Stable Decay learning rate schedules."
---

> Source: [@gm8xx8 on X](https://x.com/gm8xx8/status/2009722476612272286) · posted 2026-01-09

![Setting Batch Size for Large-Scale Pretraining](/imgs/blogs/x-bookmarks/2009722476612272286-1.jpg)

## Summary

The InternLM team at Shanghai AI Lab asks how to set the batch size for large-scale pretraining. They say the classic critical batch size treats efficiency as a fixed tradeoff between tokens and steps, but under Warmup Stable Decay that relationship breaks in the stable phase. The post is cut off at the loss curves.

## Background

Batch size is how many examples are processed per training step. The critical batch size is the point beyond which bigger batches stop saving steps. Warmup Stable Decay is a learning rate schedule with a long constant phase. Choosing batch size well saves training cost.

## What I took away

- Batch size rules of thumb may not hold under every learning rate schedule.
- Efficiency tradeoffs between tokens and steps can change across training phases.
- Read the paper for the actual findings.

## Reference

- Original post: [https://x.com/gm8xx8/status/2009722476612272286](https://x.com/gm8xx8/status/2009722476612272286)
- Author: [@gm8xx8](https://x.com/gm8xx8) on X, posted 2026-01-09
