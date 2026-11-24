---
title: "Raven and Rethinking Linear Attention States"
publishDate: "2026-08-08"
date: "2026-08-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "Reflections on a paper introducing Raven, which aims to fix memory persistence in linear models."
---

> Source: [@yffeng3920 on X](https://x.com/yffeng3920/status/2086176106340929784) · posted 2026-08-08

## Summary

The author says a paper changed how they view linear attention states and sliding-window attention, and they wrote up their reflections. They found it while searching for work on the idea of MoE of states. The quoted post says SSMs fail on recall tasks they have the capacity to solve because SSMs and sliding-window attention lack persistence, with memory either decaying or being evicted, and introduces Raven, which it says surpasses all prior linear models.

## Background

SSMs (state space models) and linear attention keep a fixed-size memory state, which makes them efficient but means old information fades. Sliding-window attention only looks at recent tokens, dropping older ones. MoE of states means having several memory states and choosing which ones to update, so some information can persist.

## What I took away

- Recall failures in linear models may come from memory design, not capacity.
- Sparse updates to multiple states are one way to keep information longer.
- Compare linear models on recall benchmarks, not only perplexity.

## Reference

- Original post: [https://x.com/yffeng3920/status/2086176106340929784](https://x.com/yffeng3920/status/2086176106340929784)
- Author: [@yffeng3920](https://x.com/yffeng3920) on X, posted 2026-08-08
