---
title: "Comment On LeapQuant Attention Gains"
publishDate: "2026-11-11"
date: "2026-11-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "llm-inference", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2120653551131402399-1.jpg"
excerpt: "Ion Stoica highlights LeapQuant, claiming near-lossless 8-bit and 4-bit quantization for attention recurrent states."
---

> Source: [@istoica05 on X](https://x.com/istoica05/status/2120653551131402399) · posted 2026-11-11

![Comment On LeapQuant Attention Gains](/imgs/blogs/x-bookmarks/2120653551131402399-1.jpg)

## Summary

The author says the work gives great performance improvements for attention with almost no downside, matching the FP32 baseline with 8-bit or 4-bit quantization. The quoted post describes **LeapQuant**, a training-free method for quantizing recurrent states in linear attention models like Qwen, Kimi and GLM.

## Background

Linear attention models carry a recurrent state that is read and rewritten every token, which is slow. Lower precision shrinks that traffic. The result is presented by the authors themselves.

## What I took away

- Low-bit state quantization is a new serving optimization.
- The claim comes from the authors' own group.
- Look for independent benchmarks.

## Reference

- Original post: [https://x.com/istoica05/status/2120653551131402399](https://x.com/istoica05/status/2120653551131402399)
- Author: [@istoica05](https://x.com/istoica05) on X, posted 2026-11-11
