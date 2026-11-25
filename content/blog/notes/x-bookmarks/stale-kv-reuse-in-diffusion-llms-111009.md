---
title: "Stale KV Reuse In Diffusion LLMs"
publishDate: "2026-06-28"
date: "2026-06-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "kv-cache"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2071312704867111009-1.jpg"
excerpt: "A post notes diffusion LLMs break KV caching through drifting keys, but quality tolerates slight drift."
---

> Source: [@nathanrs on X](https://x.com/nathanrs/status/2071312704867111009) · posted 2026-06-28

![Stale KV Reuse In Diffusion LLMs](/imgs/blogs/x-bookmarks/2071312704867111009-1.jpg)

## Summary

@nathanrs says diffusion LLM bidirectional attention makes keys and values drift across steps, breaking KV caching. Quality is robust to slight drift, and much work maximizes stale KV reuse.

## Background

A KV cache stores past keys and values to avoid recomputation. With bidirectional attention they change each step, so reuse is approximate.

## What I took away

- Approximate reuse can keep quality.
- KV caching is a key problem for diffusion LLMs.

## Reference

- Original post: [https://x.com/nathanrs/status/2071312704867111009](https://x.com/nathanrs/status/2071312704867111009)
- Author: [@nathanrs](https://x.com/nathanrs) on X, posted 2026-06-28
