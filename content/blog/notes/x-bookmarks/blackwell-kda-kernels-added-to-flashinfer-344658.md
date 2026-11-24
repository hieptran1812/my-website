---
title: "Blackwell KDA Kernels Added to FlashInfer"
publishDate: "2026-08-01"
date: "2026-08-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "A FlashInfer pull request adds optimized B200 KDA prefill kernels, claimed about 2.05x geomean faster than FlashKDA, as part of a project called CAKE."
---

> Source: [@ye_combinator on X](https://x.com/ye_combinator/status/2083415648995344658) · posted 2026-08-01

## Summary

The author announces new **Blackwell KDA kernels** in FlashInfer, with a claimed ~2.05x geometric mean speedup over FlashKDA. The work is a pull request adding an optimized B200 recurrent prefill backend. The kernels are part of something called CAKE, which the author says will be revealed later.

## Background

KDA is Kimi Delta Attention, a linear-attention style mechanism used in Kimi models. Prefill is the phase where the model processes the whole input prompt before generating tokens. FlashInfer is an open library of fast attention and inference kernels, and B200 is an NVIDIA Blackwell GPU.

## What I took away

- New attention variants need new kernels per GPU generation to be fast.
- Geomean speedup summarizes gains across many shapes; check the per-case numbers.
- Watch FlashInfer for further CAKE kernels.

## Reference

- Original post: [https://x.com/ye_combinator/status/2083415648995344658](https://x.com/ye_combinator/status/2083415648995344658)
- Author: [@ye_combinator](https://x.com/ye_combinator) on X, posted 2026-08-01
