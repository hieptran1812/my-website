---
title: "Critique Of Nemotron 3 Ultra Speed Comparison"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "quantization", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062754340428021828-1.jpg"
excerpt: "The author argues the claimed 5x speedup over Kimi K2.6 compares against a weak baseline setup."
---

> Source: [@part_harry_ on X](https://x.com/part_harry_/status/2062754340428021828) · posted 2026-06-05

![Critique Of Nemotron 3 Ultra Speed Comparison](/imgs/blogs/x-bookmarks/2062754340428021828-1.jpg)

## Summary

The author lists objections to NVIDIA's claim that Nemotron 3 Ultra is 5 times faster than Kimi K2.6. They say it likely compares to an int4 Kimi doing BF16 multiplications, whereas production would use NVFP4 Kimi. They also note TRT was used for Nemotron but vLLM for Kimi, and that Kimi is faster in TRT too.

## Background

Speed comparisons depend on precision and serving engine. An unfair baseline can inflate results, which is why people check the setup.

## What I took away

- Check precision, engine and settings before trusting speed claims.
- Compare models under the same serving stack.

## Reference

- Original post: [https://x.com/part_harry_/status/2062754340428021828](https://x.com/part_harry_/status/2062754340428021828)
- Author: [@part_harry_](https://x.com/part_harry_) on X, posted 2026-06-05
