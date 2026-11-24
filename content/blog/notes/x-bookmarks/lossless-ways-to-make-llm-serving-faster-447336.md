---
title: "Lossless Ways to Make LLM Serving Faster"
publishDate: "2026-08-03"
date: "2026-08-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "speculative-decoding", "kv-cache", "quantization"]
author: "Hiep Tran"
featured: false
excerpt: "A blog post on speeding up serving without changing the model: exact kernels, speculative decoding, KV reuse, and quantization risks."
---

> Source: [@Akashi203 on X](https://x.com/Akashi203/status/2084278896212447336) · posted 2026-08-03

## Summary

The author wrote about lossless inference, meaning making serving faster without changing the model's outputs. Topics include exact kernels, speculative decoding and KV reuse, plus the math for why quantization can quietly hurt agent runs.

## Background

Lossless methods keep outputs the same as the original model. Speculative decoding lets a small model draft tokens that the big model verifies in one pass. Quantization stores weights in fewer bits, which saves memory but can add small errors that compound over long agent tasks.

## What I took away

- Try lossless speedups before lossy ones like quantization.
- Small per-step errors can add up over long multi-step agent runs.
- Measure agent task success, not just benchmark scores, after quantizing.

## Reference

- Original post: [https://x.com/Akashi203/status/2084278896212447336](https://x.com/Akashi203/status/2084278896212447336)
- Author: [@Akashi203](https://x.com/Akashi203) on X, posted 2026-08-03
