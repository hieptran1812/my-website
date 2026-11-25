---
title: "Large Batch Size For RL"
publishDate: "2026-06-30"
date: "2026-06-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
excerpt: "A tip says very large batch sizes make RL easier, especially with noisy rewards."
---

> Source: [@hu_yifei on X](https://x.com/hu_yifei/status/2071779390300848514) · posted 2026-06-30

## Summary

@hu_yifei says a cheat code for RL is a very large batch size, using grad accumulation if needed, which works better with noisy rewards.

## Background

Larger batches average noise, giving steadier gradients. Gradient accumulation sums many small batches.

## What I took away

- Try larger batches in RL.
- Use accumulation when memory is limited.

## Reference

- Original post: [https://x.com/hu_yifei/status/2071779390300848514](https://x.com/hu_yifei/status/2071779390300848514)
- Author: [@hu_yifei](https://x.com/hu_yifei) on X, posted 2026-06-30
