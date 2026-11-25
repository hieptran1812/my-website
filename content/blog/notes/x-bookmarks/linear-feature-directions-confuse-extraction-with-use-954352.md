---
title: "Linear Feature Directions Confuse Extraction With Use"
publishDate: "2026-04-26"
date: "2026-04-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2048494068317954352-1.jpg"
excerpt: "A thread argues that finding linear feature directions in a network does not show the model actually uses them."
---

> Source: [@che_shr_cat on X](https://x.com/che_shr_cat/status/2048494068317954352) · posted 2026-04-26

![Linear Feature Directions Confuse Extraction With Use](/imgs/blogs/x-bookmarks/2048494068317954352-1.jpg)

## Summary

The author says most interpretability work assumes deep networks store features as linear directions in latent space. They call this flawed because it confuses feature extraction with feature utilization. A feature being present does not mean the model uses it.

## Background

Interpretability research tries to understand what is happening inside a neural network. A common method finds directions in the hidden space that match concepts. The point of the thread is that a direction you can read out may not be one the network relies on to compute its answer.

## What I took away

- Being able to decode a feature is not proof the model uses it.
- Causal tests, such as changing a feature and watching the output, matter more than probes alone.
- Treat interpretability claims with this gap in mind.

## Reference

- Original post: [https://x.com/che_shr_cat/status/2048494068317954352](https://x.com/che_shr_cat/status/2048494068317954352)
- Author: [@che_shr_cat](https://x.com/che_shr_cat) on X, posted 2026-04-26
