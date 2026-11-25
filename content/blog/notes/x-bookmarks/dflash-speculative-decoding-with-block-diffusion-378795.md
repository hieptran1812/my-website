---
title: "DFlash Speculative Decoding With Block Diffusion"
publishDate: "2026-01-06"
date: "2026-01-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "diffusion", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2008394269103378795-1.jpg"
excerpt: "DFlash combines block diffusion drafting with speculative decoding, claiming lossless speedups over EAGLE-3."
---

> Source: [@zhijianliu on X](https://x.com/zhijianliu/status/2008394269103378795) · posted 2026-01-06

![DFlash Speculative Decoding With Block Diffusion](/imgs/blogs/x-bookmarks/2008394269103378795-1.jpg)

## Summary

The author introduces DFlash, speculative decoding with block diffusion. They claim a 6.2x lossless speedup on Qwen3-8B and 2.5x faster than EAGLE-3. They say diffusion and autoregressive models do not have to compete, noting that dLLMs are fast and parallel but lossy.

## Background

Speculative decoding means a small model drafts several tokens and the big model checks them in one pass. A diffusion LLM (dLLM) generates many tokens in parallel rather than one at a time. Using it as the drafter keeps the big model's output exact while adding speed. Speedups are the author's claims.

## What I took away

- Parallel diffusion drafting can be paired with an autoregressive verifier to stay lossless.
- Compare against strong baselines like EAGLE-3, as the author does.
- Speedup numbers depend on model and setup; check the paper.

## Reference

- Original post: [https://x.com/zhijianliu/status/2008394269103378795](https://x.com/zhijianliu/status/2008394269103378795)
- Author: [@zhijianliu](https://x.com/zhijianliu) on X, posted 2026-01-06
