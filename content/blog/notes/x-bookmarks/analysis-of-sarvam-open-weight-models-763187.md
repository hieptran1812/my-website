---
title: "Analysis of Sarvam Open-Weight Models"
publishDate: "2026-03-07"
date: "2026-03-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2030322353641763187-1.jpg"
excerpt: "A reshare of an analysis of the Sarvam 30B and 105B models from India."
---

> Source: [@HarveenChadha on X](https://x.com/HarveenChadha/status/2030322353641763187) · posted 2026-03-07

![Analysis of Sarvam Open-Weight Models](/imgs/blogs/x-bookmarks/2030322353641763187-1.jpg)

## Summary

The author is surprised that their favorite educator analyzed the Sarvam models. The quoted post says that while waiting for DeepSeek V4, two strong open-weight reasoning LLMs came from India: Sarvam 30B and Sarvam 105B. The 30B model uses classic Grouped Query Attention (GQA), while the other model differs.

## Background

Open-weight means the model parameters are public. Grouped Query Attention shares key and value heads across several query heads to reduce memory use. Architecture comparisons like this show design choices across model sizes.

## What I took away

- Sarvam 30B and 105B are open-weight reasoning models from India.
- The 30B uses classic GQA; compare with the larger model in the full post.
- Architecture write-ups are a quick way to learn design tradeoffs.

## Reference

- Original post: [https://x.com/HarveenChadha/status/2030322353641763187](https://x.com/HarveenChadha/status/2030322353641763187)
- Author: [@HarveenChadha](https://x.com/HarveenChadha) on X, posted 2026-03-07
