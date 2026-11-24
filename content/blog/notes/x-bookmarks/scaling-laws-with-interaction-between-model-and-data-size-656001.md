---
title: "Scaling Laws With Interaction Between Model and Data Size"
publishDate: "2026-08-10"
date: "2026-08-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research", "diffusion"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher welcomes a scaling law that models how parameters and data interact, citing uses of a coupling variable."
---

> Source: [@jramapuram on X](https://x.com/jramapuram/status/2086941702015656001) · posted 2026-08-10

## Summary

The author is glad to see people coming back to the coupling variable, which their team found useful for continuous end-to-end speech diffusion at scale, tri-modal discrete diffusion (MDM) and distillation scaling laws. The quoted post links an arXiv paper with a simple scaling law formulation that considers the interaction between N and D, plus an efficient experiment sampling strategy for estimating parameters.

## Background

Scaling laws predict model loss from model size N (parameters) and data size D (tokens). Simple forms treat their effects as separate; an interaction term captures how the benefit of more data depends on model size. Good scaling laws help plan expensive training runs.

## What I took away

- Interaction terms between model and data size can make scaling predictions more accurate.
- Efficient sampling of small experiments reduces the cost of fitting scaling laws.

## Reference

- Original post: [https://x.com/jramapuram/status/2086941702015656001](https://x.com/jramapuram/status/2086941702015656001)
- Author: [@jramapuram](https://x.com/jramapuram) on X, posted 2026-08-10
