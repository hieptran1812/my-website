---
title: "RoPE Can Be Removed After Training"
publishDate: "2026-01-12"
date: "2026-01-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2010608516680036838-1.jpg"
excerpt: "Authors of DroPE find RoPE helps optimization but can be cheaply removed post-training for better length generalization."
---

> Source: [@yoav_gelberg on X](https://x.com/yoav_gelberg/status/2010608516680036838) · posted 2026-01-12

![RoPE Can Be Removed After Training](/imgs/blogs/x-bookmarks/2010608516680036838-1.jpg)

## Summary

Yoav Gelberg shares work finding that RoPE is important for optimization but can be cheaply removed after training, giving models with much better length generalization. The quoted post introduces DroPE from Sakana.

## Background

RoPE encodes position by rotating query and key vectors. Dropping it removes length limits learned during training.

## What I took away

- Position encodings may be removable after training.
- Length generalization can be cheap.

## Reference

- Original post: [https://x.com/yoav_gelberg/status/2010608516680036838](https://x.com/yoav_gelberg/status/2010608516680036838)
- Author: [@yoav_gelberg](https://x.com/yoav_gelberg) on X, posted 2026-01-12
