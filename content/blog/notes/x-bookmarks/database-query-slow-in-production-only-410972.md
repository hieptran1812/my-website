---
title: "Database Query Slow In Production Only"
publishDate: "2026-02-10"
date: "2026-02-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "career"]
author: "Hiep Tran"
featured: false
excerpt: "An interview question about a query taking 50ms locally but 2000ms in production."
---

> Source: [@Oblivious9021 on X](https://x.com/Oblivious9021/status/2021216079251410972) · posted 2026-02-10

## Summary

The post poses an interview question. A database query takes 50ms locally but 2000ms in production with the same schema and data volume, so what is happening? No answer is given.

## Background

Common causes include missing or unused indexes, different query plans, stale statistics, locking and contention, network latency, cold caches, and concurrent load. Comparing execution plans is a standard first step.

## What I took away

- Compare EXPLAIN plans between environments.
- Same data size does not mean same data distribution or load.
- Check locks, connection pools and network as well as the query.

## Reference

- Original post: [https://x.com/Oblivious9021/status/2021216079251410972](https://x.com/Oblivious9021/status/2021216079251410972)
- Author: [@Oblivious9021](https://x.com/Oblivious9021) on X, posted 2026-02-10
