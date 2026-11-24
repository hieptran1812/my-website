---
title: "Draft-Free Speculative Decoding With Medusa, EAGLE and Lookup"
publishDate: "2026-07-17"
date: "2026-07-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "An overview of speculative decoding methods that attach the drafter to the target model instead of running a separate model."
---

> Source: [@TipsCsharp on X](https://x.com/TipsCsharp/status/2078024921192538288) · posted 2026-07-17

## Summary

The post describes three flavors of draft-free speculation: Medusa, EAGLE and lookup. It notes that running a second draft model adds operational overhead, so modern methods attach the drafter to the target model. Medusa adds parallel prediction heads that guess several tokens ahead in one pass; the text cuts off before the other two.

## Background

Speculative decoding means a cheap drafter proposes several tokens and the big model checks them all in one pass, keeping the correct ones. EAGLE uses a small layer on top of the model's hidden states to draft, and lookup methods copy likely continuations from the prompt or past text.

## What I took away

- Attaching draft heads avoids deploying and syncing a separate model.
- Lookup drafting is cheap and works well when output repeats input, such as code edits.
- Speedups depend on how often drafts are accepted for your workload.

## Reference

- Original post: [https://x.com/TipsCsharp/status/2078024921192538288](https://x.com/TipsCsharp/status/2078024921192538288)
- Author: [@TipsCsharp](https://x.com/TipsCsharp) on X, posted 2026-07-17
