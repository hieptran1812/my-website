---
title: "DFlash 2 Speeds Up GLM-5.3-Flash"
publishDate: "2026-08-27"
date: "2026-08-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "DFlash 2 weights for GLM-5.3-Flash are released, claimed to make the model 2.8 times faster."
---

> Source: [@zhijianliu on X](https://x.com/zhijianliu/status/2093114089778422153) · posted 2026-08-27

## Summary

Zhijian Liu announces **DFlash 2** for **GLM-5.3-Flash**, claiming it is 2.8x faster, with weights on Hugging Face and more coming soon. The quoted post from the same team says GLM-5.3-Flash is fast and DFlash 2 makes it faster.

## Background

DFlash is a speculative decoding method. Speculative decoding means a small drafter proposes several tokens and the big model checks them in one pass, keeping only the correct ones. The output stays the same as the big model would produce, but decoding is faster.

## What I took away

- Speculative decoding drafters are now shipped as separate weights for specific models.
- Speedup depends on how often drafts are accepted, which varies by workload.
- The 2.8x figure is the team's claim; measure on your prompts.

## Reference

- Original post: [https://x.com/zhijianliu/status/2093114089778422153](https://x.com/zhijianliu/status/2093114089778422153)
- Author: [@zhijianliu](https://x.com/zhijianliu) on X, posted 2026-08-27
