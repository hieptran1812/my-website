---
title: "Optimization Work Never Ends"
publishDate: "2026-07-14"
date: "2026-07-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "A short reflection on the SGLang GLM5.2 serving work, saying optimization is endless and enjoyable."
---

> Source: [@ying11231 on X](https://x.com/ying11231/status/2077104543633166603) · posted 2026-07-14

## Summary

The author quotes the SGLang post on reaching 500+ tokens per second per user on 8xB300 for GLM5.2 NVFP4. He comments that the beauty of optimization is that it never ends, and that pushing things to the extreme is the highest form of enjoyment.

## Background

Serving performance can usually be improved through kernel tuning, scheduling, quantization and parallelism choices. Gains stack up over weeks after a model launches.

## What I took away

- Expect steady performance gains after day-0 support; re-benchmark periodically.
- See the linked SGLang deep dive for the technical details.

## Reference

- Original post: [https://x.com/ying11231/status/2077104543633166603](https://x.com/ying11231/status/2077104543633166603)
- Author: [@ying11231](https://x.com/ying11231) on X, posted 2026-07-14
