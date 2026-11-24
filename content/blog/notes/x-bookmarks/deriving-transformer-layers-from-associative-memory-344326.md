---
title: "Deriving Transformer Layers From Associative Memory"
publishDate: "2026-11-01"
date: "2026-11-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Praise for a blog post that explains transformer layers through associative memory, plus questions on gated MLPs."
---

> Source: [@ShumingHu on X](https://x.com/ShumingHu/status/2117026890389344326) · posted 2026-11-01

## Summary

The quoted blog thread asks how much models can remember and shows how you could have invented every transformer layer from the idea of memory. @ShumingHu calls it one of the best blogs he has read, praising its interactive widgets on associative memory retrieval. He also asks whether anyone has tried Gated^2-MLP or Gated^N-MLP, since gated MLPs like SwiGLU are now the default.

## Background

Associative memory stores key-value pairs and retrieves values by similarity to a query, which is closely related to attention. Gated MLPs like SwiGLU multiply two projections together, one acting as a gate, and are standard in modern LLMs.

## What I took away

- Attention and MLPs can both be seen as forms of memory lookup.
- Interactive explainers can build intuition faster than papers.
- Stacking more gates in MLPs is an open idea to explore.

## Reference

- Original post: [https://x.com/ShumingHu/status/2117026890389344326](https://x.com/ShumingHu/status/2117026890389344326)
- Author: [@ShumingHu](https://x.com/ShumingHu) on X, posted 2026-11-01
