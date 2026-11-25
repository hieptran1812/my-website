---
title: "Model Routing Can Speed Up Sessions"
publishDate: "2026-06-09"
date: "2026-06-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "serving", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2064370958417482178-1.jpg"
excerpt: "Factory says automatic model routing is faster on average because smaller models have lower latency."
---

> Source: [@factory on X](https://x.com/factory/status/2064370958417482178) · posted 2026-06-09

![Model Routing Can Speed Up Sessions](/imgs/blogs/x-bookmarks/2064370958417482178-1.jpg)

## Summary

@factory says sessions with routing are faster on average since smaller models have lower latency per token, and the routing decision is one classification call at session start.

## Background

Routing picks a model for each task. Easy tasks go to small fast models.

## What I took away

- Routing is cheap to run.
- Latency gain comes from using smaller models.

## Reference

- Original post: [https://x.com/factory/status/2064370958417482178](https://x.com/factory/status/2064370958417482178)
- Author: [@factory](https://x.com/factory) on X, posted 2026-06-09
