---
title: "DFlash 2 and Parallel Drafting for Speculative Decoding"
publishDate: "2026-09-23"
date: "2026-09-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "A talk on DFlash, which made speculative drafting parallel, and DFlash 2, which fixed the next bottleneck in token selection."
---

> Source: [@zhijianliu on X](https://x.com/zhijianliu/status/2102869385941799234) · posted 2026-09-23

## Summary

The author announces a talk at Modal Runtime about DFlash. The drafter was the serial bottleneck, so DFlash made drafting parallel; that moved the bottleneck to selection, which **DFlash 2** fixes. The quoted post claims Qwen3.8-27B runs at 70 tok/s on an M5 Max MacBook Pro, up to 4.6x faster than autoregressive decoding with the same output, and one more accepted token per pass.

## Background

Speculative decoding means a small model drafts several tokens and the big model checks them in one pass, keeping only the correct ones. If the drafter itself generates one token at a time, it becomes slow. Drafting many tokens in parallel removes that, and better selection increases how many drafted tokens are accepted.

## What I took away

- Optimization often just moves the bottleneck; look for the next one after each fix.
- Lossless speculative decoding keeps output identical while increasing speed.
- Accepted tokens per pass is a key metric for drafters.

## Reference

- Original post: [https://x.com/zhijianliu/status/2102869385941799234](https://x.com/zhijianliu/status/2102869385941799234)
- Author: [@zhijianliu](https://x.com/zhijianliu) on X, posted 2026-09-23
