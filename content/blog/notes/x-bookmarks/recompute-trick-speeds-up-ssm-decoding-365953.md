---
title: "Recompute Trick Speeds Up SSM Decoding"
publishDate: "2026-06-15"
date: "2026-06-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "speculative-decoding", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2066518563184365953-1.jpg"
excerpt: "Tri Dao says loading SSM states without storing them makes hybrid models about 2x faster and enables speculative decoding."
---

> Source: [@tri_dao on X](https://x.com/tri_dao/status/2066518563184365953) · posted 2026-06-15

![Recompute Trick Speeds Up SSM Decoding](/imgs/blogs/x-bookmarks/2066518563184365953-1.jpg)

## Summary

@tri_dao quotes the ReplaySSM post. He claims that for hybrid models running agents with huge context, Gated-DeltaNet and Mamba states are a bottleneck. Loading states, computing, and not storing them is claimed to be 2x faster and to unlock speculative decoding for SSMs.

## Background

Gated-DeltaNet and Mamba are state-space style layers with a fixed-size state. Speculative decoding needs to verify several draft tokens and roll back if wrong, which is hard if the state is overwritten each step. Recomputing the state makes rollback easy.

## What I took away

- Not storing state can make rollback cheap.
- The 2x figure is the author's claim; check the paper.
- Speculative decoding for SSMs becomes more practical.

## Reference

- Original post: [https://x.com/tri_dao/status/2066518563184365953](https://x.com/tri_dao/status/2066518563184365953)
- Author: [@tri_dao](https://x.com/tri_dao) on X, posted 2026-06-15
