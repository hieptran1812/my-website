---
title: "Trying Polymarket Spread Bots"
publishDate: "2025-12-13"
date: "2025-12-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "crypto", "trading"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1999858009426567224-1.png"
excerpt: "A developer built a Polymarket bot that buys YES and NO when the total drops below one dollar and found it harder than expected."
---

> Source: [@k1rallik on X](https://x.com/k1rallik/status/1999858009426567224) · posted 2025-12-13

![Trying Polymarket Spread Bots (1)](/imgs/blogs/x-bookmarks/1999858009426567224-1.png)

![Trying Polymarket Spread Bots (2)](/imgs/blogs/x-bookmarks/1999858009426567224-2.png)

## Summary

@k1rallik says everyone calls spread bots a gem, so they built one. It buys YES and NO on Polymarket when the sum falls below $1. They tested only 15-minute BTC, ETH, SOL and XRP markets and found it was not simple because too many bots run the same logic.

## Background

If YES and NO shares together cost less than $1, buying both locks in a payout of $1 at settlement. That is an arbitrage. The edge disappears when many bots compete for it.

## What I took away

- Arbitrage edges get crowded quickly.
- Fees and execution speed decide whether they work.
- Test simple ideas, but expect competition.

## Reference

- Original post: [https://x.com/k1rallik/status/1999858009426567224](https://x.com/k1rallik/status/1999858009426567224)
- Author: [@k1rallik](https://x.com/k1rallik) on X, posted 2025-12-13
