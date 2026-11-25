---
title: "Cross-Datacenter Prefill Decode Disaggregation With Kimi Linear"
publishDate: "2026-04-18"
date: "2026-04-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving", "kv-cache"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2045461663898599472-1.png"
excerpt: "Kimi Moonshot says it extended prefill/decode disaggregation across datacenters and mixed hardware, enabled by its hybrid Kimi Linear model."
---

> Source: [@Kimi_Moonshot on X](https://x.com/Kimi_Moonshot/status/2045461663898599472) · posted 2026-04-18

![Cross-Datacenter Prefill Decode Disaggregation With Kimi Linear](/imgs/blogs/x-bookmarks/2045461663898599472-1.png)

## Summary

Kimi Moonshot says it pushes prefill/decode disaggregation beyond one cluster, across datacenters and different hardware, to lower cost per token. KV cache transfer overhead blocked this before. The key enabler is its hybrid model, Kimi Linear.

## Background

Prefill reads the prompt and decode writes tokens one by one. They have different hardware needs, so splitting them helps. The KV cache holds attention state and moving it is costly. A hybrid model that has a smaller cache makes moving it cheaper.

## What I took away

- Smaller KV caches make splitting serving across sites practical.
- Model design and serving cost are linked.
- The lower-cost claim is the lab's own.

## Reference

- Original post: [https://x.com/Kimi_Moonshot/status/2045461663898599472](https://x.com/Kimi_Moonshot/status/2045461663898599472)
- Author: [@Kimi_Moonshot](https://x.com/Kimi_Moonshot) on X, posted 2026-04-18
