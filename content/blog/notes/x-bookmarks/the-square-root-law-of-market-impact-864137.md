---
title: "The Square Root Law of Market Impact"
publishDate: "2026-08-05"
date: "2026-08-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quant", "trading", "math"]
author: "Hiep Tran"
featured: false
excerpt: "A post explaining the square root law of market impact, which says large order price impact grows with the square root of size."
---

> Source: [@Phoenixyin13 on X](https://x.com/Phoenixyin13/status/2084867608050864137) · posted 2026-08-05

## Summary

The author calls the square root law of market impact the most complete win for the physics approach in finance. Researchers such as Bouchaud, Farmer and Toth found that the price impact of executing a large order is not linear in size, and is roughly described by the formula dP = Y * sigma * sqrt(Q/V), where Q is the order size.

## Background

Market impact is how much your own trading moves the price against you. In the formula, sigma is volatility, V is typical traded volume, Q is the order size and Y is a constant near one. Square root means doubling an order does not double the impact, which shapes how large orders are split and how strategy capacity is estimated.

## What I took away

- Impact grows with the square root of order size relative to volume, not linearly.
- Execution costs limit how much capital a strategy can handle.
- This is educational background, not a trading recommendation.

## Reference

- Original post: [https://x.com/Phoenixyin13/status/2084867608050864137](https://x.com/Phoenixyin13/status/2084867608050864137)
- Author: [@Phoenixyin13](https://x.com/Phoenixyin13) on X, posted 2026-08-05
