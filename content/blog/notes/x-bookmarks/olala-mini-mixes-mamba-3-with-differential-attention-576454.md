---
title: "Olala Mini Mixes Mamba-3 With Differential Attention"
publishDate: "2026-11-19"
date: "2026-11-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "moe", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2123495174957576454-1.jpg"
excerpt: "A post praises an experiment in a 7B model combining differential attention, Mamba-3, SSMax and GeoNorm."
---

> Source: [@_albertgu on X](https://x.com/_albertgu/status/2123495174957576454) · posted 2026-11-19

![Olala Mini Mixes Mamba-3 With Differential Attention](/imgs/blogs/x-bookmarks/2123495174957576454-1.jpg)

## Summary

The author calls it a cool experiment combining many architectural ideas including differential attention, Mamba-3, SSMax and GeoNorm. The quoted post presents Olala mini, a 7B total, 1B active model trained on 13T tokens, with a custom architecture mixing Mamba-3 and Grouped Differential Attention layers.

## Background

Mamba is a state space model, an alternative to attention that scales better with length. Differential attention subtracts two attention maps to cut noise. Hybrids try to get the best of both.

## What I took away

- Hybrid architectures are being tried at scale.
- 1B active parameters of 7B means sparse use.
- Read the architecture write-up.

## Reference

- Original post: [https://x.com/_albertgu/status/2123495174957576454](https://x.com/_albertgu/status/2123495174957576454)
- Author: [@_albertgu](https://x.com/_albertgu) on X, posted 2026-11-19
