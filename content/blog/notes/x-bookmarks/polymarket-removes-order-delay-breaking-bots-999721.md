---
title: "Polymarket Removes Order Delay Breaking Bots"
publishDate: "2026-02-23"
date: "2026-02-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "trading", "quant", "crypto"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2026008475591999721-1.jpg"
excerpt: "Author claims Polymarket removed a 500ms delay and broke latency arbitrage and polling bots."
---

> Source: [@hanakoxbt on X](https://x.com/hanakoxbt/status/2026008475591999721) · posted 2026-02-23

![Polymarket Removes Order Delay Breaking Bots (1)](/imgs/blogs/x-bookmarks/2026008475591999721-1.jpg)

![Polymarket Removes Order Delay Breaking Bots (2)](/imgs/blogs/x-bookmarks/2026008475591999721-2.jpg)

## Summary

The author claims Polymarket removed a 500ms delay without warning, and that half the bots died overnight. They say latency arbitrage and REST-polling bots now lose money on trades. The text is cut off before the author's advice.

## Background

Some trading venues add a short delay to orders, and bots built around that buffer depend on it. Removing it changes the timing assumptions of those strategies. This is an educational note on strategy fragility, not advice.

## What I took away

- Strategies built on a platform quirk can break when the platform changes.
- Monitor PnL and platform changes continuously.
- Treat the claim as the author's own, not verified.

## Reference

- Original post: [https://x.com/hanakoxbt/status/2026008475591999721](https://x.com/hanakoxbt/status/2026008475591999721)
- Author: [@hanakoxbt](https://x.com/hanakoxbt) on X, posted 2026-02-23
