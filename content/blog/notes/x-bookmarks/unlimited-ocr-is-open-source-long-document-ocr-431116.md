---
title: "Unlimited OCR Is Open-Source Long-Document OCR"
publishDate: "2026-06-23"
date: "2026-06-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2069445473627431116-1.jpg"
excerpt: "A reaction to Baidu open-sourcing Unlimited OCR, a 3B-parameter model that reads 40+ pages in one pass."
---

> Source: [@suchenzang on X](https://x.com/suchenzang/status/2069445473627431116) · posted 2026-06-23

![Unlimited OCR Is Open-Source Long-Document OCR](/imgs/blogs/x-bookmarks/2069445473627431116-1.jpg)

## Summary

@suchenzang says this is what open-source looks like. The quoted post says Unlimited OCR has 3B total and 500M activated parameters and can transcribe 40+ pages in one pass while keeping context intact.

## Background

Total parameters is the model size, while activated parameters is what runs per token in a mixture-of-experts model. Few active parameters keep inference cheap. Reading many pages in one pass preserves cross-page context.

## What I took away

- A small active size can handle long documents.
- Open release makes it easy to try.

## Reference

- Original post: [https://x.com/suchenzang/status/2069445473627431116](https://x.com/suchenzang/status/2069445473627431116)
- Author: [@suchenzang](https://x.com/suchenzang) on X, posted 2026-06-23
