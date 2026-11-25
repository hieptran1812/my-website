---
title: "Interpretability And The Bitter Lesson"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-safety", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002432869583749540-1.jpg"
excerpt: "Neel Nanda reviews papers on end-to-end interpretability, including Activation Oracles, which decode a model's own activations in natural language."
---

> Source: [@NeelNanda5 on X](https://x.com/NeelNanda5/status/2002432869583749540) · posted 2025-12-20

![Interpretability And The Bitter Lesson](/imgs/blogs/x-bookmarks/2002432869583749540-1.jpg)

## Summary

Neel Nanda posts a video asking what interpretability would look like if it embraced the bitter lesson. He reviews two papers, Activation Oracles and PCD, giving hot takes. The quoted paper trains Activation Oracles, LLMs that decode their own neural activations and answer questions about them in natural language, and finds surprising generalization, such as uncovering misaligned goals in fine-tuned models.

## Background

The bitter lesson says general methods that use more compute beat hand-built ones. End-to-end interpretability trains a model to explain activations directly rather than designing analysis tools by hand. Activations are the internal numbers a model computes.

## What I took away

- Models may be trained to explain their own internals.
- Activation Oracles generalized to finding misaligned goals without being trained for it.
- Scaling ideas may reshape interpretability research.

## Reference

- Original post: [https://x.com/NeelNanda5/status/2002432869583749540](https://x.com/NeelNanda5/status/2002432869583749540)
- Author: [@NeelNanda5](https://x.com/NeelNanda5) on X, posted 2025-12-20
