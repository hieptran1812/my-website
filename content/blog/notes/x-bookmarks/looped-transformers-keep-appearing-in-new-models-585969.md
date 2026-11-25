---
title: "Looped Transformers Keep Appearing in New Models"
publishDate: "2026-11-23"
date: "2026-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "open-models", "industry-news"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2124821870457585969-1.jpg"
excerpt: "A post notes that looped transformers are becoming common, citing GPT-6 Astra, Inkling 2 Mini and GLM-5.4."
---

> Source: [@Hesamation on X](https://x.com/Hesamation/status/2124821870457585969) · posted 2026-11-23

![Looped Transformers Keep Appearing in New Models (1)](/imgs/blogs/x-bookmarks/2124821870457585969-1.jpg)

![Looped Transformers Keep Appearing in New Models (2)](/imgs/blogs/x-bookmarks/2124821870457585969-2.jpg)

## Summary

The author says looped transformers are becoming a thing whether you like it or not, naming GPT-6 Astra, Inkling 2 Mini and now GLM-5.4. The quoted post frames the choice as scaling loop depth versus scaling model size and links the GLM-5.4 blog.

## Background

A looped transformer runs the same block of layers multiple times. This increases compute per token without adding parameters, so a model can think deeper while staying small.

## What I took away

- Looping is an alternative to adding parameters.
- Several 2026 models use it, per the author.
- Watch for trade-offs in latency and monitoring.

## Reference

- Original post: [https://x.com/Hesamation/status/2124821870457585969](https://x.com/Hesamation/status/2124821870457585969)
- Author: [@Hesamation](https://x.com/Hesamation) on X, posted 2026-11-23
