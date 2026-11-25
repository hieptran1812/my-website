---
title: "Scaling Rate Limiting Past 100K Requests"
publishDate: "2026-02-15"
date: "2026-02-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "tools", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "An interview question on scaling rate limiting when all gateways share one database."
---

> Source: [@0xlelouch_ on X](https://x.com/0xlelouch_/status/2022915193672786139) · posted 2026-02-15

## Summary

The post asks how to scale rate limiting when an API gateway hits 100K requests per second and adding instances does not help because all use the same rate-limit database.

## Background

A shared counter store becomes a bottleneck. Options include local counters with periodic sync, sharding keys, in-memory stores such as Redis clusters, and approximate algorithms.

## What I took away

- Avoid a single shared counter on the hot path.
- Accept approximate limits to gain scale.
- Shard by user or key.

## Reference

- Original post: [https://x.com/0xlelouch_/status/2022915193672786139](https://x.com/0xlelouch_/status/2022915193672786139)
- Author: [@0xlelouch_](https://x.com/0xlelouch_) on X, posted 2026-02-15
