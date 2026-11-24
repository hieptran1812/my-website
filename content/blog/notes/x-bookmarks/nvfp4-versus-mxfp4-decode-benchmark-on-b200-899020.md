---
title: "NVFP4 Versus MXFP4 Decode Benchmark on B200"
publishDate: "2026-10-11"
date: "2026-10-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "llm-inference", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "A real-workload comparison claims NVFP4 decodes up to about 8% faster than MXFP4 on B200 GPUs."
---

> Source: [@TheZachMueller on X](https://x.com/TheZachMueller/status/2109312296530899020) · posted 2026-10-11

## Summary

The author points NVFP4 fans to Stas Bekman's free ml-engineering repository on GitHub. The quoted post shares a blog benchmark by Cezar comparing NVFP4 and MXFP4 on a real inference workload. Its TLDR: on a B200 you should probably prefer NVFP4, with up to about 8% faster decode at small batches.

## Background

NVFP4 and MXFP4 are 4-bit floating point formats for weights or activations. They differ in how they group values and store scale factors, which affects both accuracy and how fast the hardware can use them.

## What I took away

- On B200, test NVFP4 first when choosing a 4-bit format.
- The gain is largest at small batch sizes, per the benchmark.
- Stas Bekman's ml-engineering repo is a free reference worth bookmarking.

## Reference

- Original post: [https://x.com/TheZachMueller/status/2109312296530899020](https://x.com/TheZachMueller/status/2109312296530899020)
- Author: [@TheZachMueller](https://x.com/TheZachMueller) on X, posted 2026-10-11
