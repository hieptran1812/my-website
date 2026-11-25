---
title: "Questions On Inkling 8.4T Tokens In 68 Hours"
publishDate: "2026-11-08"
date: "2026-11-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "quantization", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "A reaction asks what made the Inkling 2 Small NVFP4 training run so efficient."
---

> Source: [@bookwormengr on X](https://x.com/bookwormengr/status/2119229120676757956) · posted 2026-11-08

## Summary

The author is impressed and asks whether the efficiency comes from a new optimizer or a residual connection that prevents vanishing gradients. The quoted post says the main pretraining phase was 8.4T tokens in 68 hours, natively in NVFP4, one rung of a predictive scaling ladder.

## Background

NVFP4 is a 4-bit floating point format that speeds up training. A scaling ladder means testing at smaller scales to predict big runs.

## What I took away

- Native 4-bit training is feasible per the claim.
- Predictive scaling reduces risk.
- Details are not in the post.

## Reference

- Original post: [https://x.com/bookwormengr/status/2119229120676757956](https://x.com/bookwormengr/status/2119229120676757956)
- Author: [@bookwormengr](https://x.com/bookwormengr) on X, posted 2026-11-08
