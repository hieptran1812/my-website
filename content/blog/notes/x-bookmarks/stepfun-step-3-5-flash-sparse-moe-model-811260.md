---
title: "StepFun Step 3.5 Flash Sparse MoE Model"
publishDate: "2026-02-12"
date: "2026-02-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "open-models", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2021807476807811260-1.jpg"
excerpt: "A post describing StepFun's Step 3.5 Flash, a 196B sparse MoE model with 11B active parameters."
---

> Source: [@HuggingPapers on X](https://x.com/HuggingPapers/status/2021807476807811260) · posted 2026-02-12

![StepFun Step 3.5 Flash Sparse MoE Model](/imgs/blogs/x-bookmarks/2021807476807811260-1.jpg)

## Summary

The post describes **Step 3.5 Flash** from StepFun as a sparse MoE with 196B parameters and 11B active per token. It claims frontier-level reasoning comparable to GPT-5.2 xHigh and Gemini 3.0 Pro at 1/6th the decoding cost, ranking #1 on MathArena with 97.3% on AIME 2025.

## Background

In a mixture-of-experts (MoE) model, only some expert sub-networks run for each token, so active parameters are far fewer than total. This lowers cost per token while keeping large capacity. The performance numbers are the post's claims.

## What I took away

- MoE gives large capacity at lower per-token compute.
- Compare active parameters, not just total, when judging cost.
- Benchmark claims need independent checking.

## Reference

- Original post: [https://x.com/HuggingPapers/status/2021807476807811260](https://x.com/HuggingPapers/status/2021807476807811260)
- Author: [@HuggingPapers](https://x.com/HuggingPapers) on X, posted 2026-02-12
