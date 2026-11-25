---
title: "Diffusion LLMs And NVFP4 Compute Gains"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "diffusion", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062987272132444613-1.jpg"
excerpt: "A reply argues diffusion LLMs make decoding compute bound, so NVFP4 would help more than plain Q4."
---

> Source: [@PavloMolchanov on X](https://x.com/PavloMolchanov/status/2062987272132444613) · posted 2026-06-05

![Diffusion LLMs And NVFP4 Compute Gains](/imgs/blogs/x-bookmarks/2062987272132444613-1.jpg)

## Summary

The author imagines what diffusion LLMs could achieve. Decoding would be compute bound in that case, so NVFP4 should give larger gains than plain Q4 because you do not pay for dequantization. They are replying to a benchmark of NVFP4 on an RTX 5090.

## Background

Normal LLM decoding is limited mostly by memory bandwidth. Diffusion LLMs generate many tokens in parallel, which shifts the limit to raw compute. Native 4-bit math units on the chip then matter more than storage savings.

## What I took away

- Whether you are memory or compute bound decides which optimization helps.
- Native low-precision compute may suit diffusion LLMs well.
- This is the author's expectation, not a measured result.

## Reference

- Original post: [https://x.com/PavloMolchanov/status/2062987272132444613](https://x.com/PavloMolchanov/status/2062987272132444613)
- Author: [@PavloMolchanov](https://x.com/PavloMolchanov) on X, posted 2026-06-05
