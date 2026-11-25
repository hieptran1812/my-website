---
title: "Explaining Inkling 2 Mini's Attention Residual and Looping Design"
publishDate: "2026-11-20"
date: "2026-11-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "interpretability", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2123887080556384712-1.jpg"
excerpt: "A post praises a clear explanation of Inkling 2 Mini's attnres and looping architecture, with a quoted study of its inner workings."
---

> Source: [@SonglinYang4 on X](https://x.com/SonglinYang4/status/2123887080556384712) · posted 2026-11-20

![Explaining Inkling 2 Mini's Attention Residual and Looping Design](/imgs/blogs/x-bookmarks/2123887080556384712-1.jpg)

## Summary

The author thanks a researcher for a very clear explanation of Inkling 2 Mini's attnres plus looping design. The quoted post says that in looped models a layer can write to itself or read from future layers, and that with early access they studied how information flows across layers, blocks and loops.

## Background

In a looped model the same layers run repeatedly, so layers can effectively feed later passes. Studying this helps explain how such a model computes. The details are in the linked thread.

## What I took away

- Looped models have unusual information flow.
- Interpretability work can run on open weights.
- See the thread for the analysis.

## Reference

- Original post: [https://x.com/SonglinYang4/status/2123887080556384712](https://x.com/SonglinYang4/status/2123887080556384712)
- Author: [@SonglinYang4](https://x.com/SonglinYang4) on X, posted 2026-11-20
