---
title: "Redis Interview Question On Memory Limit"
publishDate: "2026-03-12"
date: "2026-03-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "career"]
author: "Hiep Tran"
featured: false
excerpt: "An interview question asking what Redis does when its in-memory data reaches the memory limit."
---

> Source: [@suni_code on X](https://x.com/suni_code/status/2031955467367563301) · posted 2026-03-12

## Summary

The post poses an interview question: if Redis stores data in memory, what happens when the memory limit is reached? The answer is not in the post text.

## Background

Redis is an in-memory key-value store. When it hits its configured memory limit, its behavior depends on an eviction policy, such as removing least recently used keys or refusing writes.

## What I took away

- Learn Redis eviction policies such as LRU and noeviction.
- Know the configured maxmemory setting and what it triggers.

## Reference

- Original post: [https://x.com/suni_code/status/2031955467367563301](https://x.com/suni_code/status/2031955467367563301)
- Author: [@suni_code](https://x.com/suni_code) on X, posted 2026-03-12
