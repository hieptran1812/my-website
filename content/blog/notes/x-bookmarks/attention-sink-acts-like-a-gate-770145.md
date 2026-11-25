---
title: "Attention Sink Acts Like A Gate"
publishDate: "2026-11-08"
date: "2026-11-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "math", "interpretability"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2119410952768770145-1.png"
excerpt: "Shows how an attention sink can be seen as a sigmoid gate on softmax-weighted values."
---

> Source: [@miyashita_03 on X](https://x.com/miyashita_03/status/2119410952768770145) · posted 2026-11-08

![Attention Sink Acts Like A Gate (1)](/imgs/blogs/x-bookmarks/2119410952768770145-1.png)

![Attention Sink Acts Like A Gate (2)](/imgs/blogs/x-bookmarks/2119410952768770145-2.png)

## Summary

The author says rearranging the softmax shows an attention sink can implicitly act as a gate: a sigmoid multiplied by the usual softmax-weighted values. Gated attention generalizes it with element-wise gating. The quoted post found attn_sink and gated_attn give almost the same validation loss.

## Background

An attention sink is a token that absorbs attention mass so other tokens are not forced to attend. Gating gives the model an explicit way to ignore output. The two ideas overlap mathematically.

## What I took away

- Sinks and gates are related.
- They may give similar loss.
- Use better eval sets such as long context.

## Reference

- Original post: [https://x.com/miyashita_03/status/2119410952768770145](https://x.com/miyashita_03/status/2119410952768770145)
- Author: [@miyashita_03](https://x.com/miyashita_03) on X, posted 2026-11-08
