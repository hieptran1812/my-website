---
title: "Expert Parallelism Mesh Limits In Distributed Training"
publishDate: "2026-06-02"
date: "2026-06-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "pretraining", "gpu-kernels"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2061850328572506355-1.jpg"
excerpt: "A note that a 1-D parallelism mesh limits expert parallelism, and DP ranks are often repurposed as EP."
---

> Source: [@nrehiew_ on X](https://x.com/nrehiew_/status/2061850328572506355) · posted 2026-06-02

![Expert Parallelism Mesh Limits In Distributed Training (1)](/imgs/blogs/x-bookmarks/2061850328572506355-1.jpg)

![Expert Parallelism Mesh Limits In Distributed Training (2)](/imgs/blogs/x-bookmarks/2061850328572506355-2.png)

![Expert Parallelism Mesh Limits In Distributed Training (3)](/imgs/blogs/x-bookmarks/2061850328572506355-3.jpg)

![Expert Parallelism Mesh Limits In Distributed Training (4)](/imgs/blogs/x-bookmarks/2061850328572506355-4.jpg)

## Summary

The author adds to a talk on distributed training. A one-dimensional DP x TP x PP x EP mesh limits the degree of EP, since EP=64 blows up world size. A common fix is repurposing DP ranks as EP so no separate dimension is needed, though the cut-off post notes a problem when EP equals DP.

## Background

Data parallel (DP), tensor parallel (TP), pipeline parallel (PP) and expert parallel (EP) split work across GPUs in different ways. Multiplying them together grows the number of GPUs needed.

## What I took away

- Parallelism dimensions multiply the world size.
- Reusing DP ranks for EP is common but has trade-offs.

## Reference

- Original post: [https://x.com/nrehiew_/status/2061850328572506355](https://x.com/nrehiew_/status/2061850328572506355)
- Author: [@nrehiew_](https://x.com/nrehiew_) on X, posted 2026-06-02
