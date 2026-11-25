---
title: "Attention Sink Biases In Xiaomi MiMo-V2-Flash"
publishDate: "2025-12-17"
date: "2025-12-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "moe", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001296943134507422-1.jpg"
excerpt: "gu_xiangming is glad attention sink biases work well, in reaction to Xiaomi MiMo-V2-Flash, an open MoE model."
---

> Source: [@gu_xiangming on X](https://x.com/gu_xiangming/status/2001296943134507422) · posted 2025-12-17

![Attention Sink Biases In Xiaomi MiMo-V2-Flash](/imgs/blogs/x-bookmarks/2001296943134507422-1.jpg)

## Summary

The author says they are glad attention sink biases work in language modeling and long-context settings and points to their post and a talk. The quoted post introduces Xiaomi MiMo-V2-Flash, an open-source MoE with 309B total and 15B active parameters, hybrid attention with 5:1 interleaved 128-window sliding window and global layers, and 256K context.

## Background

An attention sink is a token that attention heads can dump unused attention onto, which keeps sliding-window attention stable. Sliding window attention only looks at recent tokens, which saves memory. Mixing it with some global layers keeps long-range ability. MoE means only some parameters are active per token.

## What I took away

- Attention sinks help stability with windowed attention.
- A 5:1 mix of window and global layers is a way to cut long-context cost.
- Active parameters (15B) matter for speed more than total (309B).

## Reference

- Original post: [https://x.com/gu_xiangming/status/2001296943134507422](https://x.com/gu_xiangming/status/2001296943134507422)
- Author: [@gu_xiangming](https://x.com/gu_xiangming) on X, posted 2025-12-17
