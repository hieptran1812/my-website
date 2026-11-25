---
title: "Mistral 3 Large Uses DeepSeek V3 Architecture"
publishDate: "2025-12-12"
date: "2025-12-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "moe", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1999557830131494977-1.jpg"
excerpt: "An observation that Mistral 3 Large appears to use the DeepSeek V3 architecture, including MLA, with different expert counts."
---

> Source: [@rasbt on X](https://x.com/rasbt/status/1999557830131494977) · posted 2025-12-12

![Mistral 3 Large Uses DeepSeek V3 Architecture](/imgs/blogs/x-bookmarks/1999557830131494977-1.jpg)

## Summary

@rasbt went through the config files and found Mistral 3 Large uses the DeepSeek V3 architecture, including MLA. The only difference he could see is that it uses 2x fewer experts but each expert is 2x larger.

## Background

MLA is multi-head latent attention, a design that shrinks the KV cache. DeepSeek V3 is a mixture-of-experts model, where each token goes to a few experts. Reusing an open architecture shows how designs spread across labs.

## What I took away

- Config files reveal how models are built.
- Open architectures get reused by other labs.
- Expert count and size is a tunable trade-off.

## Reference

- Original post: [https://x.com/rasbt/status/1999557830131494977](https://x.com/rasbt/status/1999557830131494977)
- Author: [@rasbt](https://x.com/rasbt) on X, posted 2025-12-12
