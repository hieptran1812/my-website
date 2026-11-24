---
title: "Low Hanging Fruit in Harness Engineering"
publishDate: "2026-07-30"
date: "2026-07-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "harness-engineering", "ai-agents", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "Nathan Lambert says harness engineering has large easy wins, citing a reported 188% score jump for GPT-5.6 Sol with harness changes."
---

> Source: [@natolambert on X](https://x.com/natolambert/status/2082667019045081511) · posted 2026-07-30

## Summary

Nathan Lambert says there is a lot of easy gain in **harness engineering**, and studying harness effects from post-training through evaluation and inference is an important area. He expects big cost savings per unit of performance. The quoted post says that implementing the harness with the Responses API and turning on retained reasoning and context compaction raised GPT-5.6 Sol's public-set score by 188% while using 6x fewer output tokens.

## Background

A harness is the code that wraps a model to run tasks: it manages context, tools and reasoning state. Retained reasoning means keeping the model's earlier reasoning between turns instead of dropping it. Context compaction summarizes or trims old context so long tasks fit.

## What I took away

- Before changing models, try improving the harness; gains can be large.
- Evaluation scores depend on the harness, so compare models under the same setup.
- Retained reasoning and context compaction are concrete features to test.

## Reference

- Original post: [https://x.com/natolambert/status/2082667019045081511](https://x.com/natolambert/status/2082667019045081511)
- Author: [@natolambert](https://x.com/natolambert) on X, posted 2026-07-30
