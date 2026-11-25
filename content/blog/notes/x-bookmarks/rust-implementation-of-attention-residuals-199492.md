---
title: "Rust Implementation Of Attention Residuals"
publishDate: "2026-03-16"
date: "2026-03-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "open-models", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2033567301627199492-1.jpg"
excerpt: "The author congratulates the Kimi team on Attention Residuals and shares a Rust implementation built with Burn."
---

> Source: [@AbdelStark on X](https://x.com/AbdelStark/status/2033567301627199492) · posted 2026-03-16

![Rust Implementation Of Attention Residuals (1)](/imgs/blogs/x-bookmarks/2033567301627199492-1.jpg)

![Rust Implementation Of Attention Residuals (2)](/imgs/blogs/x-bookmarks/2033567301627199492-2.jpg)

## Summary

The author praises the Kimi Moonshot team's research and says they built a Rust implementation with full AttnRes and Block AttnRes plus two-phase inference, using Burn. The quoted post introduces Attention Residuals, which rethink depth-wise aggregation by replacing standard depth-wise recurrence.

## Background

Residual connections add a layer's input to its output with fixed weights. Attention Residuals instead let the model choose how to combine earlier layers, inspired by attention over time. Burn is a Rust deep learning framework.

## What I took away

- Residual connections are open to redesign.
- Re-implementing a paper in another language is a good way to learn it.

## Reference

- Original post: [https://x.com/AbdelStark/status/2033567301627199492](https://x.com/AbdelStark/status/2033567301627199492)
- Author: [@AbdelStark](https://x.com/AbdelStark) on X, posted 2026-03-16
