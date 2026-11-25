---
title: "Nonparametric Skew From Mean Minus Median"
publishDate: "2026-03-21"
date: "2026-03-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "math", "quant"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2035343399163760792-1.png"
excerpt: "The author explains a skew measure, (mean - median) / stdev, bounded between -1 and 1 and robust to outliers."
---

> Source: [@VivekVRao1 on X](https://x.com/VivekVRao1/status/2035343399163760792) · posted 2026-03-21

![Nonparametric Skew From Mean Minus Median](/imgs/blogs/x-bookmarks/2035343399163760792-1.png)

## Summary

The author says (mean(x) - median(x)) / stdev(x) is a nonparametric skew measure bounded in (-1, 1), and is less sensitive to outliers than the moment-based estimator. They suggest using it for rolling skew on returns data. The quoted post notes that |mean - median| is at most the standard deviation and gives a direct proof.

## Background

Skew describes how lopsided a distribution is. The usual formula uses the third power of deviations, so one extreme value can dominate it. Mean minus median uses robust pieces, and the bound follows from the quoted inequality.

## What I took away

- For any variable, |mean - median| <= stdev.
- This gives a bounded, outlier-resistant skew measure.
- Consider it for rolling skew on noisy returns.

## Reference

- Original post: [https://x.com/VivekVRao1/status/2035343399163760792](https://x.com/VivekVRao1/status/2035343399163760792)
- Author: [@VivekVRao1](https://x.com/VivekVRao1) on X, posted 2026-03-21
