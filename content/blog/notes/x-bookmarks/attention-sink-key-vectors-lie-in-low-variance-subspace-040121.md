---
title: "Attention Sink Key Vectors Lie In Low-Variance Subspace"
publishDate: "2025-11-21"
date: "2025-11-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1991721857545040121-1.jpg"
excerpt: "A research finding that key vectors of attention sink tokens are squashed into a low-variance subspace."
---

> Source: [@HeMuyu0327 on X](https://x.com/HeMuyu0327/status/1991721857545040121) · posted 2025-11-21

![Attention Sink Key Vectors Lie In Low-Variance Subspace (1)](/imgs/blogs/x-bookmarks/1991721857545040121-1.jpg)

![Attention Sink Key Vectors Lie In Low-Variance Subspace (2)](/imgs/blogs/x-bookmarks/1991721857545040121-2.jpg)

![Attention Sink Key Vectors Lie In Low-Variance Subspace (3)](/imgs/blogs/x-bookmarks/1991721857545040121-3.jpg)

## Summary

The authors say it was known that the value vectors of attention sinks are small. They now look at the key vectors, which directly contribute to the sink, and find that sink-token keys are squashed into a low-variance subspace that the query token can easily align with.

## Background

An attention sink is a token, often the first one, that gets a lot of attention without carrying much meaning. It acts as a place to dump attention weight. Understanding how keys and queries form sinks helps explain model behavior and can guide efficient attention designs.

## What I took away

- Sinks work through both small values and structured keys.
- Low-variance key subspaces make sinks easy for queries to find.
- Check the thread for the full finding.

## Reference

- Original post: [https://x.com/HeMuyu0327/status/1991721857545040121](https://x.com/HeMuyu0327/status/1991721857545040121)
- Author: [@HeMuyu0327](https://x.com/HeMuyu0327) on X, posted 2025-11-21
