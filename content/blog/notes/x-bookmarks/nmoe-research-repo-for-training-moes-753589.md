---
title: "nmoe Research Repo For Training MoEs"
publishDate: "2025-12-17"
date: "2025-12-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "pretraining", "llm-architecture", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001223632861753589-1.jpg"
excerpt: "_xjdr previews nmoe, a small research-focused repo for quick MoE experiments with pluggable attention types."
---

> Source: [@_xjdr on X](https://x.com/_xjdr/status/2001223632861753589) · posted 2025-12-17

![nmoe Research Repo For Training MoEs](/imgs/blogs/x-bookmarks/2001223632861753589-1.jpg)

## Summary

The author posts a first update and a code preview of nmoe from Noumena Network. The quoted post, Why Training MoEs is So Hard, says he wanted a small research-focused training repo for fast experiments, including pluggable attention architectures such as MLA, SWA, NSA and KDA. A detailed post will follow.

## Background

MoE (mixture of experts) models use many expert sub-networks and activate only some per token, which makes training harder than for dense models. A small codebase with swappable attention modules makes it easier to compare ideas quickly. MLA, SWA, NSA and KDA are different attention designs.

## What I took away

- A small hackable repo speeds up architecture experiments.
- Pluggable attention modules make fair comparisons easier.
- Read the code while waiting for the detailed write-up.

## Reference

- Original post: [https://x.com/_xjdr/status/2001223632861753589](https://x.com/_xjdr/status/2001223632861753589)
- Author: [@_xjdr](https://x.com/_xjdr) on X, posted 2025-12-17
