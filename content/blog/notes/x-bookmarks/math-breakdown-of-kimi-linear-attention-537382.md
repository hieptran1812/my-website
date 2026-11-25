---
title: "Math Breakdown Of Kimi Linear Attention"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "math", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002245655352537382-1.jpg"
excerpt: "HeMuyu0327 derives how Kimi Delta Attention improves on basic linear attention, step by step."
---

> Source: [@HeMuyu0327 on X](https://x.com/HeMuyu0327/status/2002245655352537382) · posted 2025-12-20

![Math Breakdown Of Kimi Linear Attention (1)](/imgs/blogs/x-bookmarks/2002245655352537382-1.jpg)

![Math Breakdown Of Kimi Linear Attention (2)](/imgs/blogs/x-bookmarks/2002245655352537382-2.jpg)

## Summary

The author breaks down the math of Kimi linear attention (KDA) to show how it evolves from the most basic linear attention. They describe linear attention from two perspectives, starting with a linear fast memory view. The post is a thread that walks through the derivation.

## Background

Standard attention costs grow with the square of the sequence length. Linear attention rewrites it as a running memory that is updated token by token, which makes the cost grow linearly. Variants like KDA change how that memory is updated and forgotten to improve quality.

## What I took away

- Linear attention can be seen as a fast-updating memory matrix.
- Understanding the basic version makes later variants like KDA easier.
- Work through the derivation by hand to see where the gains come from.

## Reference

- Original post: [https://x.com/HeMuyu0327/status/2002245655352537382](https://x.com/HeMuyu0327/status/2002245655352537382)
- Author: [@HeMuyu0327](https://x.com/HeMuyu0327) on X, posted 2025-12-20
