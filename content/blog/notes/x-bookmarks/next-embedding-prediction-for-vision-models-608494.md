---
title: "Next-Embedding Prediction For Vision Models"
publishDate: "2025-12-22"
date: "2025-12-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002987708306608494-1.jpg"
excerpt: "A post announces NEPA, which trains vision models by predicting the next embedding instead of reconstructing pixels."
---

> Source: [@jiqizhixin on X](https://x.com/jiqizhixin/status/2002987708306608494) · posted 2025-12-22

![Next-Embedding Prediction For Vision Models](/imgs/blogs/x-bookmarks/2002987708306608494-1.jpg)

## Summary

The post presents Next-Embedding Prediction (NEPA) from researchers at U. Michigan, NYU, Princeton and U. Virginia. Instead of reconstructing pixels, the model learns by predicting the next embedding in a visual sequence. The author claims this lets vision models be trained much like ChatGPT-style language models.

## Background

An embedding is a vector of numbers that represents an image patch. Language models learn by predicting the next token, and this approach applies the same next-step idea to visual embeddings. If it works, vision and language training could share one simple recipe.

## What I took away

- Next-step prediction is being applied beyond text.
- Predicting embeddings avoids pixel reconstruction losses.
- The 'train like ChatGPT' framing is the author's claim; check the paper.

## Reference

- Original post: [https://x.com/jiqizhixin/status/2002987708306608494](https://x.com/jiqizhixin/status/2002987708306608494)
- Author: [@jiqizhixin](https://x.com/jiqizhixin) on X, posted 2025-12-22
