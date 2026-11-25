---
title: "Interview Question On Serving Llama-3 To Many Users"
publishDate: "2025-12-06"
date: "2025-12-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving", "kv-cache"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1997292199005057370-1.jpg"
excerpt: "An interview-style post asking how to serve Llama-3 to 1,000 users and why memory runs out even if the model fits."
---

> Source: [@asmah2107 on X](https://x.com/asmah2107/status/1997292199005057370) · posted 2025-12-06

![Interview Question On Serving Llama-3 To Many Users](/imgs/blogs/x-bookmarks/1997292199005057370-1.jpg)

## Summary

@asmah2107 poses a Meta ML interview question: how do you serve Llama-3 to 1,000 concurrent users, and why do you run out of memory even if the model fits on the GPU? The thread then gives the author's answer, not included in the text.

## Background

Besides model weights, each active request stores a KV cache, the attention keys and values for its tokens. With many users and long contexts, the KV cache can exceed the memory left after loading weights. Serving systems use batching and cache management to cope.

## What I took away

- Weights are only part of GPU memory use; the KV cache grows with users and length.
- Memory planning for concurrency is a core serving skill.

## Reference

- Original post: [https://x.com/asmah2107/status/1997292199005057370](https://x.com/asmah2107/status/1997292199005057370)
- Author: [@asmah2107](https://x.com/asmah2107) on X, posted 2025-12-06
