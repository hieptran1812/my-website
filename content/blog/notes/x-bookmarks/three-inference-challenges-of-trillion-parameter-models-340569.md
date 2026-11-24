---
title: "Three Inference Challenges of Trillion-Parameter Models"
publishDate: "2026-10-19"
date: "2026-10-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "moe", "llm-architecture", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A post shares a diagram explaining width, depth and length challenges in serving trillion-parameter models."
---

> Source: [@BrianLi23 on X](https://x.com/BrianLi23/status/2112294787676340569) · posted 2026-10-19

## Summary

@BrianLi23 calls the content a masterclass. The quoted post says scaling past a trillion parameters requires inference engineers to solve three issues: width (routing between hundreds of sparse experts), depth (mixing global and local linear attention), and length (processing 1M-token sequences), and that two people drew it up.

## Background

Mixture-of-experts models have many expert sub-networks and route each token to a few, which creates communication challenges. Linear and local attention reduce the cost of attention over long inputs. Very long sequences stress memory through the KV cache.

## What I took away

- Think of large-model serving as three axes: experts, attention mix and sequence length.
- Each axis needs different engineering techniques.
- Open the link to see the diagram.

## Reference

- Original post: [https://x.com/BrianLi23/status/2112294787676340569](https://x.com/BrianLi23/status/2112294787676340569)
- Author: [@BrianLi23](https://x.com/BrianLi23) on X, posted 2026-10-19
