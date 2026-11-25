---
title: "How JSON Parsing Costs CPU Time"
publishDate: "2025-11-22"
date: "2025-11-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "tools", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A post arguing that JSON text parsing is a hidden CPU cost that can slow APIs."
---

> Source: [@techNmak on X](https://x.com/techNmak/status/1992288304134570423) · posted 2025-11-22

## Summary

The author says a slow API may not be the database or network, but the language of the data. JSON is just a text string, and every time you send something like an id field, the server pays a hidden parsing cost.

## Background

Parsing text into in-memory objects needs CPU work, and large or frequent payloads add up. Binary formats such as Protocol Buffers avoid some of that cost. The post makes this case, though real gains depend on your workload.

## What I took away

- Measure serialization cost before blaming the database.
- Binary formats can reduce parsing overhead at high volume.
- Profile first, since JSON is often fine at small scale.

## Reference

- Original post: [https://x.com/techNmak/status/1992288304134570423](https://x.com/techNmak/status/1992288304134570423)
- Author: [@techNmak](https://x.com/techNmak) on X, posted 2025-11-22
