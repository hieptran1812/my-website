---
title: "Scaling LLM Inference With CPU KV Cache Offloading"
publishDate: "2026-08-28"
date: "2026-08-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Day 20 of an inference infra series runs a real deployment experiment on how CPU KV offloading improves serving."
---

> Source: [@jaga_prasanna on X](https://x.com/jaga_prasanna/status/2093220764241772834) · posted 2026-08-28

## Summary

Day 20 of the author's Inference Infra Series covers **KV offloading at scale**, using a real deployment and experiment to see how CPU KV offloading improves serving. The quoted post is titled The Complete Guide to Scaling LLM Inference with CPU Offloading.

## Background

The KV cache stores attention keys and values for past tokens so the model does not recompute them. It uses lots of GPU memory. Offloading moves some of it to cheaper, larger CPU memory, so more requests or longer prefixes can be kept and reused instead of recomputed.

## What I took away

- CPU memory can act as an overflow tier for the KV cache.
- Offloading helps most when prefixes are reused, such as in multi-turn chats and agents.
- Transfer bandwidth between CPU and GPU limits how much offloading helps.

## Reference

- Original post: [https://x.com/jaga_prasanna/status/2093220764241772834](https://x.com/jaga_prasanna/status/2093220764241772834)
- Author: [@jaga_prasanna](https://x.com/jaga_prasanna) on X, posted 2026-08-28
