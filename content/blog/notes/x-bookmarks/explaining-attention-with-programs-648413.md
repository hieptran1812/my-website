---
title: "Explaining Attention With Programs"
publishDate: "2026-06-29"
date: "2026-06-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2071735685275648413-1.jpg"
excerpt: "A paper explains attention heads using Python programs, replacing ~40% in Llama-3B with little loss."
---

> Source: [@belindazli on X](https://x.com/belindazli/status/2071735685275648413) · posted 2026-06-29

![Explaining Attention With Programs (1)](/imgs/blogs/x-bookmarks/2071735685275648413-1.jpg)

![Explaining Attention With Programs (2)](/imgs/blogs/x-bookmarks/2071735685275648413-2.jpg)

## Summary

@belindazli introduces automated interpretability where heads are explained by Python programs, with ~40% of attention patterns in Llama-3B replaceable with barely affected performance. The quoted post says attention can be human-readable code.

## Background

An attention head's pattern describes which tokens it attends to. If code matches it, the head is understood.

## What I took away

- Heads can be summarized as code.
- Check arXiv 2606.19317.

## Reference

- Original post: [https://x.com/belindazli/status/2071735685275648413](https://x.com/belindazli/status/2071735685275648413)
- Author: [@belindazli](https://x.com/belindazli) on X, posted 2026-06-29
