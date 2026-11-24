---
title: "Interactive Tutorial on Speculative Decoding"
publishDate: "2026-09-06"
date: "2026-09-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "learning-resources", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "A summary of an interactive NeurIPS tutorial on how speculative decoding evolved and when it stays lossless."
---

> Source: [@di_zhang_fdu on X](https://x.com/di_zhang_fdu/status/2096682973676663261) · posted 2026-09-06

## Summary

The author explains that speculative decoding speeds up LLM serving: a small draft model proposes several tokens and the target model verifies them. Strict rejection sampling keeps the output distribution the same as the target model. They share the original post introducing an interactive tutorial for the NeurIPS Education Track.

## Background

LLMs generate one token at a time, which underuses the GPU. Verifying several drafted tokens in one pass is cheaper than generating each separately. Rejection sampling decides which drafts to accept so the final output matches what the big model alone would produce.

## What I took away

- Speculative decoding can be lossless when rejection sampling is done strictly.
- Use the interactive tutorial to build intuition.
- Relaxed acceptance rules trade exactness for speed.

## Reference

- Original post: [https://x.com/di_zhang_fdu/status/2096682973676663261](https://x.com/di_zhang_fdu/status/2096682973676663261)
- Author: [@di_zhang_fdu](https://x.com/di_zhang_fdu) on X, posted 2026-09-06
