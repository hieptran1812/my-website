---
title: "NVIDIA Transformer Variant With a Fourth Projection"
publishDate: "2026-09-06"
date: "2026-09-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "llm-inference", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "NVIDIA researchers add a fourth projection to attention layers, reported to speed decoding 1.7x and improve long reasoning accuracy."
---

> Source: [@DailyDoseOfDS_ on X](https://x.com/DailyDoseOfDS_/status/2096531369669394532) · posted 2026-09-06

## Summary

The post says NVIDIA researchers made a new transformer variant with one small change. It reports 1.7x faster decoding and long-reasoning accuracy up 6.5 points. Normally each attention layer computes Q, K and V, and NVIDIA's tweak adds a fourth projection; the rest is cut off.

## Background

In attention, Q (query), K (key) and V (value) are linear projections of the input; queries are matched against keys to weight the values. Changing these projections can reduce KV cache size or computation, which speeds up decoding.

## What I took away

- Small attention changes can have large effects on decode speed.
- Read the paper to see what the fourth projection does.
- This is a summary account's description, so check the source.

## Reference

- Original post: [https://x.com/DailyDoseOfDS_/status/2096531369669394532](https://x.com/DailyDoseOfDS_/status/2096531369669394532)
- Author: [@DailyDoseOfDS_](https://x.com/DailyDoseOfDS_) on X, posted 2026-09-06
