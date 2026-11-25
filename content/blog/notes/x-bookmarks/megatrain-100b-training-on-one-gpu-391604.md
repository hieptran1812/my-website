---
title: "MegaTrain 100B Training On One GPU"
publishDate: "2026-06-29"
date: "2026-06-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "pretraining", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2071383986677391604-1.jpg"
excerpt: "A post introduces MegaTrain, claiming full-precision training of 100B+ LLMs on a single GPU."
---

> Source: [@che_shr_cat on X](https://x.com/che_shr_cat/status/2071383986677391604) · posted 2026-06-29

![MegaTrain 100B Training On One GPU](/imgs/blogs/x-bookmarks/2071383986677391604-1.jpg)

## Summary

@che_shr_cat says GPU memory is treated wrongly and MegaTrain enables full-precision training of 100B+ LLMs on one GPU by making VRAM a transient stateless cache and inverting the memory hierarchy.

## Background

VRAM is GPU memory. Offloading to host memory trades speed for capacity.

## What I took away

- Memory hierarchy can be inverted.
- Claims need verification.

## Reference

- Original post: [https://x.com/che_shr_cat/status/2071383986677391604](https://x.com/che_shr_cat/status/2071383986677391604)
- Author: [@che_shr_cat](https://x.com/che_shr_cat) on X, posted 2026-06-29
