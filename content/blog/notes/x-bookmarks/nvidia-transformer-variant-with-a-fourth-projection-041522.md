---
title: "NVIDIA Transformer Variant With a Fourth Projection"
publishDate: "2026-08-12"
date: "2026-08-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "llm-inference", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "NVIDIA researchers add a fourth projection to attention layers, reportedly speeding decoding and improving reasoning accuracy."
---

> Source: [@akshay_pachaar on X](https://x.com/akshay_pachaar/status/2087519615132041522) · posted 2026-08-12

## Summary

The author reports that NVIDIA researchers built a new transformer variant where one small change made decoding 1.7 times faster and raised long-reasoning accuracy by 6.5 points. In a typical transformer, every attention layer computes Q, K and V, and NVIDIA's tweak adds a fourth projection. The post is cut off before describing it fully.

## Background

In attention, each token is projected into a query (Q), key (K) and value (V); queries are matched against keys to decide which values to combine. Adding another projection changes what each layer can compute or store. The speed and accuracy numbers are the post's claims and should be checked in the paper.

## What I took away

- Small architectural changes in attention can have large effects on speed and quality.
- Read the paper to see what the fourth projection does and its costs.
- Treat headline improvements as claims until replicated.

## Reference

- Original post: [https://x.com/akshay_pachaar/status/2087519615132041522](https://x.com/akshay_pachaar/status/2087519615132041522)
- Author: [@akshay_pachaar](https://x.com/akshay_pachaar) on X, posted 2026-08-12
