---
title: "Deep Dive on Paged Attention and KV Cache"
publishDate: "2026-02-03"
date: "2026-02-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference", "serving", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2018736429480788239-1.jpg"
excerpt: "A detailed write-up on Paged Attention and KV cache management in inference systems like vLLM, from first principles."
---

> Source: [@hamzaelshafie on X](https://x.com/hamzaelshafie/status/2018736429480788239) · posted 2026-02-03

![Deep Dive on Paged Attention and KV Cache (1)](/imgs/blogs/x-bookmarks/2018736429480788239-1.jpg)

![Deep Dive on Paged Attention and KV Cache (2)](/imgs/blogs/x-bookmarks/2018736429480788239-2.jpg)

![Deep Dive on Paged Attention and KV Cache (3)](/imgs/blogs/x-bookmarks/2018736429480788239-3.jpg)

![Deep Dive on Paged Attention and KV Cache (4)](/imgs/blogs/x-bookmarks/2018736429480788239-4.jpg)

## Summary

The author wrote an in-depth breakdown of Paged Attention and KV cache management in systems like vLLM. It starts with training versus inference, prefill versus decoding, why KV caching exists, and where memory fragmentation comes from, then moves to vLLM-style approaches.

## Background

The KV cache stores attention keys and values for past tokens so they are not recomputed. It grows with each token and fragments memory. Paged Attention stores it in fixed-size blocks like virtual memory pages, wasting less space.

## What I took away

- KV cache memory is the main limit on batch size.
- Paging reduces fragmentation.
- Good starting point to learn LLM serving.

## Reference

- Original post: [https://x.com/hamzaelshafie/status/2018736429480788239](https://x.com/hamzaelshafie/status/2018736429480788239)
- Author: [@hamzaelshafie](https://x.com/hamzaelshafie) on X, posted 2026-02-03
