---
title: "Polymarket Volume Double-Counted On Dashboards"
publishDate: "2025-12-08"
date: "2025-12-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "crypto", "trading", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1998127491609923969-1.jpg"
excerpt: "A claim that most major dashboards double-count Polymarket volume because its onchain data repeats each trade."
---

> Source: [@notnotstorm on X](https://x.com/notnotstorm/status/1998127491609923969) · posted 2025-12-08

![Polymarket Volume Double-Counted On Dashboards (1)](/imgs/blogs/x-bookmarks/1998127491609923969-1.jpg)

![Polymarket Volume Double-Counted On Dashboards (2)](/imgs/blogs/x-bookmarks/1998127491609923969-2.png)

## Summary

@notnotstorm says they found a major data bug: almost every major dashboard has been double-counting Polymarket volume. The author says this is separate from wash trading and is caused by redundant representations of each trade in Polymarket's onchain data. The post promises receipts in a follow-up.

## Background

Onchain data is the public record of transactions on a blockchain. If one trade shows up as more than one record, a dashboard that adds all records will report volume that is too high. Wash trading means fake trades to inflate volume, which is a different issue.

## What I took away

- Check how a dashboard counts data before trusting headline volume.
- Duplicate records are a common source of inflated metrics.
- This is the author's claim, supported by evidence in the thread.

## Reference

- Original post: [https://x.com/notnotstorm/status/1998127491609923969](https://x.com/notnotstorm/status/1998127491609923969)
- Author: [@notnotstorm](https://x.com/notnotstorm) on X, posted 2025-12-08
