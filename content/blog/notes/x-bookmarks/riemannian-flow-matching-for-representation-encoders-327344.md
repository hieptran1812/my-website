---
title: "Riemannian Flow Matching For Representation Encoders"
publishDate: "2026-02-19"
date: "2026-02-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research", "math"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2024531993011327344-1.jpg"
excerpt: "Sid Dancha highlights work using spherical geometry to train diffusion transformers on features like DINOv2."
---

> Source: [@siddancha on X](https://x.com/siddancha/status/2024531993011327344) · posted 2026-02-19

![Riemannian Flow Matching For Representation Encoders](/imgs/blogs/x-bookmarks/2024531993011327344-1.jpg)

## Summary

The author notes that DINOv2-style representations trained with contrastive losses lie on a high-dimensional sphere. So instead of straight-line flow matching, one should use Riemannian flow matching on that manifold. The quoted paper says standard DiTs fail on such features because of geometry, not just parameter count.

## Background

Flow matching trains a model to move noise to data along a path. Straight paths ignore that features live on a sphere. Following the curved surface keeps samples valid and eases training.

## What I took away

- Respect the geometry of your data when designing paths.
- Failure to converge can be a geometry issue, not a size issue.

## Reference

- Original post: [https://x.com/siddancha/status/2024531993011327344](https://x.com/siddancha/status/2024531993011327344)
- Author: [@siddancha](https://x.com/siddancha) on X, posted 2026-02-19
