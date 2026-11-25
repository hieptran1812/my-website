---
title: "Data Amount And Memorization Saturation"
publishDate: "2026-07-03"
date: "2026-07-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2073113247532892491-1.jpg"
excerpt: "A TLDR says do not train LLMs on too little data or image models for too many epochs."
---

> Source: [@gabriberton on X](https://x.com/gabriberton/status/2073113247532892491) · posted 2026-07-03

![Data Amount And Memorization Saturation](/imgs/blogs/x-bookmarks/2073113247532892491-1.jpg)

## Summary

@gabriberton gives a TLDR: avoid training LLMs on too little data and image generators for too many epochs. The quoted post says LLMs memorize until saturation at about 3.6 bits per parameter, after which they compress, generalize, show double descent and grok.

## Background

Memorization capacity is limited per parameter. When data exceeds it, the model must generalize. Too little data lets it memorize instead.

## What I took away

- A model has finite memorization capacity per parameter.
- Past saturation, generalization takes over.
- Match data amount to model size.

## Reference

- Original post: [https://x.com/gabriberton/status/2073113247532892491](https://x.com/gabriberton/status/2073113247532892491)
- Author: [@gabriberton](https://x.com/gabriberton) on X, posted 2026-07-03
