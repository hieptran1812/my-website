---
title: "Three Foundational Papers for LLM Inference"
publishDate: "2026-07-08"
date: "2026-07-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "learning-resources", "kv-cache"]
author: "Hiep Tran"
featured: false
excerpt: "A recommendation of three papers to master LLM inference, starting with PagedAttention from vLLM."
---

> Source: [@Alacritic_Super on X](https://x.com/Alacritic_Super/status/2074810528158093547) · posted 2026-07-08

## Summary

The author recommends three papers for mastering LLM inference, saying they introduced many ideas behind today's fastest engines. The first is PagedAttention (vLLM), arXiv 2309.06180, which matters for efficient KV cache memory management. The rest of the list is cut off.

## Background

PagedAttention stores the KV cache in fixed-size blocks, like virtual memory pages in an operating system, so memory is not wasted on padding and many requests can share GPU memory. This made high-throughput batching practical.

## What I took away

- Start inference learning with PagedAttention, since many engines build on it.
- Open the post to see the other two papers.
- Reading foundational papers explains why engine features exist.

## Reference

- Original post: [https://x.com/Alacritic_Super/status/2074810528158093547](https://x.com/Alacritic_Super/status/2074810528158093547)
- Author: [@Alacritic_Super](https://x.com/Alacritic_Super) on X, posted 2026-07-08
