---
title: "Exa Dynamic Highlights Cut Agent Context Usage"
publishDate: "2026-08-28"
date: "2026-08-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "tools", "harness-engineering"]
author: "Hiep Tran"
featured: false
excerpt: "Exa releases dynamic highlights, a model that filters redundant and irrelevant page content, claimed to cut token usage 95% on average."
---

> Source: [@yoimnotkesku on X](https://x.com/yoimnotkesku/status/2093423023399305469) · posted 2026-08-28

## Summary

The author says agents waste a lot of context when given entire pages and that their team fixed it. The quoted post announces a research preview of **dynamic highlights** from Exa: given a query and a set of pages, a custom model scans all of them together to remove redundant and irrelevant information, reducing token usage by 95% on average.

## Background

Agents that browse the web often paste whole pages into the model's context, which is costly and can distract the model. Filtering content to the parts relevant to the query keeps context small. Scanning pages jointly also removes content repeated across sources.

## What I took away

- Feeding whole pages to agents wastes tokens and can hurt answers.
- Query-aware filtering before the main model is a cheap improvement to try.
- The 95% figure is Exa's average claim; measure on your own queries.

## Reference

- Original post: [https://x.com/yoimnotkesku/status/2093423023399305469](https://x.com/yoimnotkesku/status/2093423023399305469)
- Author: [@yoimnotkesku](https://x.com/yoimnotkesku) on X, posted 2026-08-28
