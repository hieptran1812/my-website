---
title: "NanoGPT Speedrun Record With a Huge Embedding"
publishDate: "2026-09-29"
date: "2026-09-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A new NanoGPT speedrun record of 39.9s uses a 65B embedding table for a 124M model, optimizing at the individual flop level."
---

> Source: [@YouJiacheng on X](https://x.com/YouJiacheng/status/2104809524121215285) · posted 2026-09-29

## Summary

The quoted post reports a new NanoGPT speedrun record of 39.9 seconds from @DevenPzak, down from 67.6 seconds. It says the record optimizes at the individual flop level instead of optimizing matmuls or adding new operations. The author reacts to one detail: a 65B embedding for a 124M model.

## Background

The NanoGPT speedrun is a community competition to train a small GPT-2 style model to a target loss as fast as possible. An embedding table maps tokens to vectors and costs memory but very little compute per token, since you only look up rows. A huge embedding is a way to add parameters without adding many flops.

## What I took away

- Parameters and compute are not the same; lookups add capacity cheaply.
- Speedrun records are a good source of practical training tricks.
- Counting flops at a fine level can reveal savings that kernel tuning misses.

## Reference

- Original post: [https://x.com/YouJiacheng/status/2104809524121215285](https://x.com/YouJiacheng/status/2104809524121215285)
- Author: [@YouJiacheng](https://x.com/YouJiacheng) on X, posted 2026-09-29
