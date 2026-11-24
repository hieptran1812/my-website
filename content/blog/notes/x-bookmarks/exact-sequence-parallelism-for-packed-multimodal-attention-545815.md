---
title: "Exact Sequence Parallelism for Packed Multimodal Attention"
publishDate: "2026-10-19"
date: "2026-10-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-architecture", "pretraining"]
author: "Hiep Tran"
featured: false
excerpt: "Luma shares a sequence parallelism method that picks ring attention, sparse all-to-all or Ulysses per layer for packed multimodal masks."
---

> Source: [@thomasneff93 on X](https://x.com/thomasneff93/status/2112228198939545815) · posted 2026-10-19

## Summary

@thomasneff93 says his team worked on a fast and exact sequence parallelism strategy and found no single strategy is best, but found a strong overall approach through evaluations and kernel work. The quoted Luma post says the method chooses ring attention, a sparse all-to-all or Ulysses per layer, matches single-GPU numerics, and beats a FlashMask plus Ulysses hybrid in 47 of 48 cases on up to 128 H100s.

## Background

Sequence parallelism splits a long input across GPUs so each handles part of it. Ring attention passes chunks around GPUs in a loop; Ulysses swaps data so each GPU handles some attention heads for the full sequence. Packed multimodal masks mean many samples of text and images share one sequence, making attention patterns irregular.

## What I took away

- The best parallelism strategy can differ per layer and per mask pattern.
- Exact methods that match single-GPU numerics avoid hidden training drift.
- Read the Luma post for the selection logic.

## Reference

- Original post: [https://x.com/thomasneff93/status/2112228198939545815](https://x.com/thomasneff93/status/2112228198939545815)
- Author: [@thomasneff93](https://x.com/thomasneff93) on X, posted 2026-10-19
