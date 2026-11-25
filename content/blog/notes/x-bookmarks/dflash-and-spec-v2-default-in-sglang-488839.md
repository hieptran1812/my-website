---
title: "DFlash And Spec V2 Default In SGLang"
publishDate: "2026-06-15"
date: "2026-06-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2066567734398488839-1.jpg"
excerpt: "SGLang makes DFlash plus Spec V2 its default speculative decoding, claiming over 4.3x baseline throughput."
---

> Source: [@radixark on X](https://x.com/radixark/status/2066567734398488839) · posted 2026-06-15

![DFlash And Spec V2 Default In SGLang](/imgs/blogs/x-bookmarks/2066567734398488839-1.jpg)

## Summary

@radixark says with Modal and z-lab.ai they made over 4.3x throughput the new default in SGLang. The quoted blog says DFlash + Spec V2 hits over 4.3x baseline throughput and is now the default speculative decoding engine, with thanks to the integration authors.

## Background

Speculative decoding drafts tokens with a small model and verifies them in one pass. Throughput is tokens per second across users. The 4.3x figure is the authors' claim.

## What I took away

- Speculative decoding is becoming a default.
- Update SGLang to benefit.

## Reference

- Original post: [https://x.com/radixark/status/2066567734398488839](https://x.com/radixark/status/2066567734398488839)
- Author: [@radixark](https://x.com/radixark) on X, posted 2026-06-15
