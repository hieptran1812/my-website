---
title: "Jacobi Forcing Turns AR LLMs Into Parallel Decoders"
publishDate: "2025-12-17"
date: "2025-12-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "diffusion"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001409875516211558-1.jpg"
excerpt: "Hao Zhang introduces Jacobi Forcing, a recipe that makes autoregressive LLMs causal parallel decoders with about 4x speedup."
---

> Source: [@haozhangml on X](https://x.com/haozhangml/status/2001409875516211558) · posted 2025-12-17

![Jacobi Forcing Turns AR LLMs Into Parallel Decoders](/imgs/blogs/x-bookmarks/2001409875516211558-1.jpg)

## Summary

Hao Zhang presents Jacobi Forcing, a recipe that turns any autoregressive LLM into a native causal parallel decoder. The quoted post says it trains AR models as diffusion-style parallel decoders with a 4x speedup while staying causal and keeping generation quality high. These are the authors' claims.

## Background

Autoregressive models generate one token at a time. Diffusion LLMs generate many in parallel but change the architecture. Jacobi decoding guesses several tokens and refines them until they stop changing, and training can teach a model to converge faster. This keeps the causal structure and the existing model.

## What I took away

- You may get parallel decoding without switching to a diffusion architecture.
- The 4x speedup is a reported claim to verify.
- Keeping causality helps reuse existing serving stacks.

## Reference

- Original post: [https://x.com/haozhangml/status/2001409875516211558](https://x.com/haozhangml/status/2001409875516211558)
- Author: [@haozhangml](https://x.com/haozhangml) on X, posted 2025-12-17
