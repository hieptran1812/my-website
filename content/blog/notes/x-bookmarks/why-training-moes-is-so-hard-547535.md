---
title: "Why Training MoEs Is So Hard"
publishDate: "2025-12-07"
date: "2025-12-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "pretraining", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1997459906719547535-1.jpg"
excerpt: "A write-up on why training mixture-of-experts models is hard, motivated by a small research-focused training repo."
---

> Source: [@_xjdr on X](https://x.com/_xjdr/status/1997459906719547535) · posted 2025-12-07

![Why Training MoEs Is So Hard](/imgs/blogs/x-bookmarks/1997459906719547535-1.jpg)

## Summary

@_xjdr shares a post titled Why Training MoEs Is So Hard. They wanted a small research-focused training repo for quick experiments with pluggable attention architectures such as MLA, SWA, NSA and KDA. The text is cut off before the main reasons.

## Background

A mixture-of-experts (MoE) model routes each token to a few specialized sub-networks, so only part of the model is used per token. MoEs are efficient at inference but tricky to train because of load balancing and communication. MLA, SWA, NSA and KDA are attention variants.

## What I took away

- MoE training has extra difficulties compared with dense models.
- A small, pluggable repo makes architecture experiments faster.

## Reference

- Original post: [https://x.com/_xjdr/status/1997459906719547535](https://x.com/_xjdr/status/1997459906719547535)
- Author: [@_xjdr](https://x.com/_xjdr) on X, posted 2025-12-07
