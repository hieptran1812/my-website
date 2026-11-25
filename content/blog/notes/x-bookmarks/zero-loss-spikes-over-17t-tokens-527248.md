---
title: "Zero Loss Spikes Over 17T Tokens"
publishDate: "2026-06-29"
date: "2026-06-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "moe"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2071663232721527248-1.jpg"
excerpt: "A post praises a paper showing no loss spikes across 17T tokens of training a 400B MoE."
---

> Source: [@gurtej__gill_ on X](https://x.com/gurtej__gill_/status/2071663232721527248) · posted 2026-06-29

![Zero Loss Spikes Over 17T Tokens](/imgs/blogs/x-bookmarks/2071663232721527248-1.jpg)

## Summary

@gurtej__gill_ calls the training loss curve wild, noting that scaling a 400B MoE usually brings stability trouble, so zero loss spikes across 17T tokens is impressive. The post adds that Muon optimizer is proving its worth (text cut off).

## Background

A loss spike is a sudden jump in training loss that can derail a run. Muon is an optimizer alternative to Adam. Stability at scale saves compute.

## What I took away

- Training stability at scale is hard to achieve.
- Muon is worth watching.
- Details come from the paper.

## Reference

- Original post: [https://x.com/gurtej__gill_/status/2071663232721527248](https://x.com/gurtej__gill_/status/2071663232721527248)
- Author: [@gurtej__gill_](https://x.com/gurtej__gill_) on X, posted 2026-06-29
