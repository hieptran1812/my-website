---
title: "Xiaomi MOPD Multi-Teacher Distillation"
publishDate: "2026-06-30"
date: "2026-06-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2071895743871197546-1.jpg"
excerpt: "A post covers Xiaomi's MOPD paper on combining multiple teachers, now an industry standard baseline."
---

> Source: [@teortaxesTex on X](https://x.com/teortaxesTex/status/2071895743871197546) · posted 2026-06-30

![Xiaomi MOPD Multi-Teacher Distillation (1)](/imgs/blogs/x-bookmarks/2071895743871197546-1.jpg)

![Xiaomi MOPD Multi-Teacher Distillation (2)](/imgs/blogs/x-bookmarks/2071895743871197546-2.jpg)

![Xiaomi MOPD Multi-Teacher Distillation (3)](/imgs/blogs/x-bookmarks/2071895743871197546-3.png)

## Summary

@teortaxesTex says Xiaomi wrote a paper on MOPD used for MiMo, similar to what DS and GLM do, framed as capability integration, with Mix-RL second best. The quoted post says OPD combines multiple teachers and token-level KL works well enough.

## Background

On-policy distillation (OPD) trains a student on its own samples while teachers score tokens. Merging several specialist teachers gives one model many skills. KL is a distance between distributions.

## What I took away

- Multi-teacher OPD is a baseline now.
- Token-level KL may be enough.
- Mix-RL is a close second.

## Reference

- Original post: [https://x.com/teortaxesTex/status/2071895743871197546](https://x.com/teortaxesTex/status/2071895743871197546)
- Author: [@teortaxesTex](https://x.com/teortaxesTex) on X, posted 2026-06-30
