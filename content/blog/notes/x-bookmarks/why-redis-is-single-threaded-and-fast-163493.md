---
title: "Why Redis Is Single-Threaded And Fast"
publishDate: "2025-11-19"
date: "2025-11-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "tools", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A post explaining why Redis uses one core for commands and is still very fast."
---

> Source: [@techNmak on X](https://x.com/techNmak/status/1991214988481163493) · posted 2025-11-19

## Summary

The post asks why Redis seems to use only one core on a 32-core server yet stays among the fastest in-memory databases. It starts a numbered explanation, beginning with the idea that the real bottleneck is usually not the CPU.

## Background

Redis runs commands on one thread, which avoids locks and context switching. Since data is in memory, the main limits are often network and memory rather than compute. You scale it by running multiple instances.

## What I took away

- Single-threaded design can avoid locking overhead.
- Network I/O is often the real bottleneck.
- Scale Redis by sharding across instances.

## Reference

- Original post: [https://x.com/techNmak/status/1991214988481163493](https://x.com/techNmak/status/1991214988481163493)
- Author: [@techNmak](https://x.com/techNmak) on X, posted 2025-11-19
