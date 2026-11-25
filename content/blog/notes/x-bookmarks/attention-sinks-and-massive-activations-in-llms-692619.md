---
title: "Attention Sinks and Massive Activations in LLMs"
publishDate: "2026-02-02"
date: "2026-02-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2018332011157692619-1.jpg"
excerpt: "A researcher discusses how outlier activations rescale dimensions to create attention sinks, replying to a post on outliers and normalization."
---

> Source: [@gu_xiangming on X](https://x.com/gu_xiangming/status/2018332011157692619) · posted 2026-02-02

![Attention Sinks and Massive Activations in LLMs (1)](/imgs/blogs/x-bookmarks/2018332011157692619-1.jpg)

![Attention Sinks and Massive Activations in LLMs (2)](/imgs/blogs/x-bookmarks/2018332011157692619-2.jpg)

![Attention Sinks and Massive Activations in LLMs (3)](/imgs/blogs/x-bookmarks/2018332011157692619-3.jpg)

## Summary

The author considers writing a blog on attention sink circuits. They think massive activations act as a rescaling of non-outlier dimensions to give a key bias and then an attention sink. The quoted post says outliers exist because with normalization they scale other dimensions.

## Background

Attention sinks are tokens that attract much attention regardless of content. Massive activations are very large values in a few dimensions. Normalization makes them useful to the model.

## What I took away

- Outliers may serve a purpose.
- Giving the model a scaling mechanism may remove them.
- Relevant to quantization where outliers hurt.

## Reference

- Original post: [https://x.com/gu_xiangming/status/2018332011157692619](https://x.com/gu_xiangming/status/2018332011157692619)
- Author: [@gu_xiangming](https://x.com/gu_xiangming) on X, posted 2026-02-02
