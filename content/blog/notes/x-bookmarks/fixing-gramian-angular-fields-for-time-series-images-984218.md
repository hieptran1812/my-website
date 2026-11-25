---
title: "Fixing Gramian Angular Fields For Time Series Images"
publishDate: "2025-11-29"
date: "2025-11-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "math"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1994791741161984218-1.jpg"
excerpt: "A note on turning time series into images with GAF and fixing the arccos ambiguity with multiple channels."
---

> Source: [@Zeeskylaw on X](https://x.com/Zeeskylaw/status/1994791741161984218) · posted 2025-11-29

![Fixing Gramian Angular Fields For Time Series Images](/imgs/blogs/x-bookmarks/1994791741161984218-1.jpg)

## Summary

The author explains that Gramian Angular Fields (GAF) turn time series into images encoding the relation between every pair of time points. Classic GAF has a flaw because arccos is symmetric, so different values can map to the same angle. Their implementation fixes this using a multi-channel approach. The link card is titled Encoding Times series as Images: GAF Imagification & The Sepsis Texture.

## Background

Turning a time series into an image lets standard computer vision models process it. GAF scales values, converts them to angles, and builds a matrix from angle sums or differences. Ambiguity can lose information.

## What I took away

- Imaging time series lets you reuse vision models.
- Watch for information loss in encodings.

## Reference

- Original post: [https://x.com/Zeeskylaw/status/1994791741161984218](https://x.com/Zeeskylaw/status/1994791741161984218)
- Author: [@Zeeskylaw](https://x.com/Zeeskylaw) on X, posted 2025-11-29
