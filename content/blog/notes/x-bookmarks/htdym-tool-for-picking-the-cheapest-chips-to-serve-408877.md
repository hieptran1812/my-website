---
title: "HTDYM Tool for Picking the Cheapest Chips to Serve Models"
publishDate: "2026-08-27"
date: "2026-08-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "hardware", "serving", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "Sail Research built HTDYM, a tool that analytically finds which chips and shardings serve a model most cheaply."
---

> Source: [@transmissions11 on X](https://x.com/transmissions11/status/2093010629854408877) · posted 2026-08-27

## Summary

The author says they worked on making TPUs fast this summer at Sail Research and built **HTDYM**, a tool that works out analytically which chips and shardings can serve a model the cheapest. They note the answer is not the same for every model or workload.

## Background

Sharding means splitting a model across several chips, for example by layers or by tensors. The best hardware and sharding depend on model size, memory needs and request patterns. An analytical tool estimates cost from these factors without running every setup.

## What I took away

- The cheapest serving hardware depends on the model and workload.
- Model cost analytically before renting chips.
- TPUs can be competitive for serving when configured well.

## Reference

- Original post: [https://x.com/transmissions11/status/2093010629854408877](https://x.com/transmissions11/status/2093010629854408877)
- Author: [@transmissions11](https://x.com/transmissions11) on X, posted 2026-08-27
