---
title: "Attention Sink Biases Work In MiMo-V2-Flash"
publishDate: "2025-12-17"
date: "2025-12-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "open-models", "moe"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001296153128939597-1.png"
excerpt: "gu_xiangming celebrates attention sink biases working well, quoting Xiaomi's MiMo-V2-Flash engineering notes on hybrid sliding window attention."
---

> Source: [@gu_xiangming on X](https://x.com/gu_xiangming/status/2001296153128939597) · posted 2025-12-17

![Attention Sink Biases Work In MiMo-V2-Flash (1)](/imgs/blogs/x-bookmarks/2001296153128939597-1.png)

![Attention Sink Biases Work In MiMo-V2-Flash (2)](/imgs/blogs/x-bookmarks/2001296153128939597-2.jpg)

![Attention Sink Biases Work In MiMo-V2-Flash (3)](/imgs/blogs/x-bookmarks/2001296153128939597-3.jpg)

## Summary

The author is glad to see attention biases work well in language modeling and long context, and says he has long advocated attention sinks, especially with shifted window attention. He links his slides. The quoted post is the MiMo-V2-Flash team's notes on engineering choices, including a Hybrid SWA architecture they found simple, elegant and better than other linear options in internal benchmarks.

## Background

Sliding window attention (SWA) limits each token to recent tokens, saving compute and memory. An attention sink is a bias that gives heads a place to put extra attention weight so windowed attention stays stable. Hybrid designs combine SWA with a few global layers.

## What I took away

- Hybrid SWA can beat more complex linear attention in the team's tests.
- Attention sinks are a small change that help windowed models.
- Read the MiMo notes for the engineering choices that mattered.

## Reference

- Original post: [https://x.com/gu_xiangming/status/2001296153128939597](https://x.com/gu_xiangming/status/2001296153128939597)
- Author: [@gu_xiangming](https://x.com/gu_xiangming) on X, posted 2025-12-17
