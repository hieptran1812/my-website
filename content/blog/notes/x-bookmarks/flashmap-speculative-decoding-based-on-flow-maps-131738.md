---
title: "FlashMap Speculative Decoding Based on Flow Maps"
publishDate: "2026-10-20"
date: "2026-10-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "diffusion"]
author: "Hiep Tran"
featured: false
excerpt: "Authors announce FlashMap, a speculative decoding algorithm using flow maps that claims over 6x faster inference on Qwen3-8B."
---

> Source: [@jeff_shen_ on X](https://x.com/jeff_shen_/status/2112379880390131738) · posted 2026-10-20

## Summary

@jeff_shen_ announces FlashMap, a new speculative decoding algorithm based on flow maps, claiming more than 6x faster inference on Qwen3-8B. The post links a project page and arXiv paper and lists co-authors.

## Background

Speculative decoding means a small model drafts several tokens and the big model checks them in one pass, so output matches the big model but comes faster. Flow maps can generate several tokens with dependencies between them in one or two passes, making them a good drafter.

## What I took away

- Better drafters raise the number of accepted tokens per verification.
- The 6x figure is the authors' claim for one model.
- See the sedielem note in this batch for expert commentary.

## Reference

- Original post: [https://x.com/jeff_shen_/status/2112379880390131738](https://x.com/jeff_shen_/status/2112379880390131738)
- Author: [@jeff_shen_](https://x.com/jeff_shen_) on X, posted 2026-10-20
