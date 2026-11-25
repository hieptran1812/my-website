---
title: "Shard KV Cache Compression Claim on Llama-3.1-8B"
publishDate: "2026-05-25"
date: "2026-05-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "quantization", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2059041521576648980-1.jpg"
excerpt: "Krish Garg claims his Shard method gets 10x KV cache compression with no quality loss, beating TurboQuant."
---

> Source: [@krishgarg on X](https://x.com/krishgarg/status/2059041521576648980) · posted 2026-05-25

![Shard KV Cache Compression Claim on Llama-3.1-8B](/imgs/blogs/x-bookmarks/2059041521576648980-1.jpg)

## Summary

The author claims to have beaten Google DeepMind's TurboQuant with **Shard**, saying it gives 10x KV cache compression on Llama-3.1-8B with zero quality loss: 10x at 8K context and 11.2x at 32K, NIAH recall 1.000 across 4K-32K and LongBench delta near 0 versus FP16. He says TurboQuant tops out at 4-6x.

## Background

The KV cache stores past attention keys and values and grows with context length, so compressing it saves memory. NIAH tests retrieving a fact buried in long text. These are the author's self-reported numbers and need independent checking.

## What I took away

- KV cache memory is a main limit for long context serving.
- Zero loss at 10x is a strong claim; look for independent replication.
- Test on tasks beyond needle retrieval.

## Reference

- Original post: [https://x.com/krishgarg/status/2059041521576648980](https://x.com/krishgarg/status/2059041521576648980)
- Author: [@krishgarg](https://x.com/krishgarg) on X, posted 2026-05-25
