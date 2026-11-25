---
title: "Gemma 4 Architecture Recap"
publishDate: "2026-04-02"
date: "2026-04-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2039751171556954531-1.jpg"
excerpt: "Elie Bakouch lists subtle architecture details of Google's Gemma 4 models."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2039751171556954531) · posted 2026-04-02

![Gemma 4 Architecture Recap (1)](/imgs/blogs/x-bookmarks/2039751171556954531-1.jpg)

![Gemma 4 Architecture Recap (2)](/imgs/blogs/x-bookmarks/2039751171556954531-2.jpg)

## Summary

The author recaps Gemma 4 architecture: per-layer embeddings only on the small variant, no attention scale (no division by sqrt(d)), QK norm plus V norm, and some sharing. The quoted post announces Gemma 4 in 31B, 26B A4B, E2B and E4B sizes under Apache 2.0.

## Background

QK norm normalizes queries and keys before attention for stability. Normally attention scores are divided by sqrt(d); skipping that is unusual. A4B suggests a mixture-of-experts model with about 4B active parameters.

## What I took away

- Small architecture tweaks differ between model families.
- Normalization choices affect training stability.
- Apache 2.0 licensing makes Gemma 4 usable commercially.

## Reference

- Original post: [https://x.com/eliebakouch/status/2039751171556954531](https://x.com/eliebakouch/status/2039751171556954531)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-04-02
