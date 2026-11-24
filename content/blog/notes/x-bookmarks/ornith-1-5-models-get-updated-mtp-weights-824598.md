---
title: "Ornith-1.5 Models Get Updated MTP Weights"
publishDate: "2026-08-24"
date: "2026-08-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "speculative-decoding", "quantization"]
author: "Hiep Tran"
featured: false
excerpt: "Ornith-1.5 open models receive updated multi-token prediction weights for BF16, GGUF, NVFP4 and FP8 variants."
---

> Source: [@ornith_ on X](https://x.com/ornith_/status/2092002467269824598) · posted 2026-08-24

## Summary

Ornith says the **Ornith-1.5** models have updated MTP weights for BF16, GGUF, NVFP4 and FP8 variants, making tokens faster. The quoted launch post introduces Ornith-1.5 as an open-source family of 9B dense, 35B MoE and 397B MoE models trained with self-improving strategies, claiming state-of-the-art results among open models of similar size and performance comparable to Claude Opus, where the text is cut off.

## Background

MTP (multi-token prediction) adds heads that predict several future tokens, which can be used as a built-in drafter for speculative decoding. BF16, FP8 and NVFP4 are number formats of decreasing precision, and GGUF is a file format for llama.cpp; lower precision saves memory and speed.

## What I took away

- MTP heads give speculative decoding without a separate draft model.
- Providing many quantized variants makes a model usable on more hardware.
- Comparisons to closed models are the authors' claims until independently tested.

## Reference

- Original post: [https://x.com/ornith_/status/2092002467269824598](https://x.com/ornith_/status/2092002467269824598)
- Author: [@ornith_](https://x.com/ornith_) on X, posted 2026-08-24
