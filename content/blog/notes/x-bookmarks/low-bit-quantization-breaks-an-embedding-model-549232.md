---
title: "Low-Bit Quantization Breaks An Embedding Model"
publishDate: "2026-04-13"
date: "2026-04-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "llm-inference", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2043503663843549232-1.jpg"
excerpt: "Han Xiao shows that low-bit quantization of jina-v5 makes embeddings lose discriminative power."
---

> Source: [@hxiao on X](https://x.com/hxiao/status/2043503663843549232) · posted 2026-04-13

![Low-Bit Quantization Breaks An Embedding Model](/imgs/blogs/x-bookmarks/2043503663843549232-1.jpg)

## Summary

The author says low quant weights make the embedding model lose all discriminative power. Their cosine correlation matrix for jina-v5 shows high off-diagonal similarities at Q1, Q2 and Q3, meaning everything looks similar.

## Background

Quantization stores weights with fewer bits to save memory. An embedding model turns text into vectors, and similar text should be close. If all vectors look alike, search stops working.

## What I took away

- Test quantized embeddings, not just LLMs.
- Plot similarity matrices to check.
- Low bits can destroy quality.

## Reference

- Original post: [https://x.com/hxiao/status/2043503663843549232](https://x.com/hxiao/status/2043503663843549232)
- Author: [@hxiao](https://x.com/hxiao) on X, posted 2026-04-13
