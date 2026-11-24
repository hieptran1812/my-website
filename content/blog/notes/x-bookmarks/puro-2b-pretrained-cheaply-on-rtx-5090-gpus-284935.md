---
title: "Puro-2B Pretrained Cheaply on RTX 5090 GPUs"
publishDate: "2026-08-30"
date: "2026-08-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "open-models", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "Puro-2B is a 2B model with a fully open recipe, claimed to match Qwen2-1.5B when trained on RTX 5090s for under $5090."
---

> Source: [@openhonor on X](https://x.com/openhonor/status/2093994169618284935) · posted 2026-08-30

## Summary

The post asks how much it costs to pretrain a 2B LLM from scratch and announces **Puro-2B** with a fully open recipe. The authors claim you can train a model matching Qwen2-1.5B on RTX 5090s for less than $5090: $4.4K to beat Qwen2-1.5B and $6.9K to approach Qwen2.5-1.5B.

## Background

Pretraining is the first, most expensive stage where a model learns from a large text corpus. It is usually done on data center GPUs. Showing it can be done on consumer RTX 5090 cards with an open recipe lowers the barrier for small teams and students.

## What I took away

- Small-model pretraining is now within reach of individual budgets, per the authors.
- Open recipes let you reproduce and learn from the full training process.
- Compare against the specific baselines named, since newer models are much stronger.

## Reference

- Original post: [https://x.com/openhonor/status/2093994169618284935](https://x.com/openhonor/status/2093994169618284935)
- Author: [@openhonor](https://x.com/openhonor) on X, posted 2026-08-30
