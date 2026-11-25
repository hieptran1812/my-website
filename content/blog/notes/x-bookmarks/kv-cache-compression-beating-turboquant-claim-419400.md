---
title: "KV Cache Compression Beating TurboQuant Claim"
publishDate: "2026-04-06"
date: "2026-04-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "quantization", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "The author claims 5.95x KV cache compression by exploiting low-dimensional signal, beating TurboQuant."
---

> Source: [@ashwingop on X](https://x.com/ashwingop/status/2041118359891419400) · posted 2026-04-06

## Summary

The author says only 4 of 128 dimensions in the KV cache carry real signal and the other 124 are noise. He claims this structure gives **5.95x compression** versus TurboQuant's 5.02x, plus 4.5x faster attention.

## Background

The KV cache stores attention keys and values for past tokens so the model does not recompute them; it grows with context length and eats GPU memory. Compression shrinks it so longer contexts or more users fit. These numbers are the author's claim and need independent checking.

## What I took away

- The KV cache can be very compressible if its signal sits in few dimensions.
- Compression ratio alone is not enough; check quality and speed too.
- Treat headline benchmarks as claims until reproduced.

## Reference

- Original post: [https://x.com/ashwingop/status/2041118359891419400](https://x.com/ashwingop/status/2041118359891419400)
- Author: [@ashwingop](https://x.com/ashwingop) on X, posted 2026-04-06
