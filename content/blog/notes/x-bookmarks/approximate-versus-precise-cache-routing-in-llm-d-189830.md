---
title: "Approximate Versus Precise Cache Routing in llm-d"
publishDate: "2026-08-06"
date: "2026-08-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "kv-cache", "llm-inference", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Day 15 of an inference infrastructure series explains how the llm-d router picks GPU pods based on which pod holds a prompt's prefix."
---

> Source: [@jaga_prasanna on X](https://x.com/jaga_prasanna/status/2085267645516189830) · posted 2026-08-06

## Summary

Day 15 of the author's 30-day inference infrastructure series covers approximate versus precise cache routing. The endpoint picker (EPP) in the llm-d router chooses the best GPU pod for each prompt, and to do so it must know which pod holds which prefix in its cache. The quoted day 14 post covered fault tolerance in disaggregated serving, such as propagating cancellations and handling prefill worker crashes.

## Background

Prefix caching keeps the KV cache for shared prompt beginnings, such as a system prompt, so repeated requests skip that work. Routing a request to the pod that already has its prefix cached saves compute. Approximate routing guesses cache contents, while precise routing tracks them exactly at higher bookkeeping cost.

## What I took away

- Cache-aware routing can matter as much as kernel speed in a serving cluster.
- There is a trade-off between routing precision and tracking overhead.
- Disaggregated serving needs explicit handling of cancellations and worker failures.

## Reference

- Original post: [https://x.com/jaga_prasanna/status/2085267645516189830](https://x.com/jaga_prasanna/status/2085267645516189830)
- Author: [@jaga_prasanna](https://x.com/jaga_prasanna) on X, posted 2026-08-06
