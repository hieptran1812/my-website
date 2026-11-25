---
title: "Unsupervised Recovery of Feature Geometry"
publishDate: "2026-05-21"
date: "2026-05-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2057508157854007515-1.jpg"
excerpt: "CFGeek suggests recovering feature geometry directly from activations instead of the two-step SAE then clustering approach."
---

> Source: [@CFGeek on X](https://x.com/CFGeek/status/2057508157854007515) · posted 2026-05-21

![Unsupervised Recovery of Feature Geometry](/imgs/blogs/x-bookmarks/2057508157854007515-1.jpg)

## Summary

The author says the next step is to directly and without supervision recover feature geometry from activations, rather than the two-step sparse autoencoder then clustering pipeline. The quoted post describes clustering SAE features by firing patterns so a cluster of features reveals the overall geometry.

## Background

Sparse autoencoders (SAEs) break model activations into features. Clustering features that fire together shows shapes in the representation. The author suggests a more direct method.

## What I took away

- Features interact, so look at groups, not single features.
- Two-step pipelines may be replaced by direct methods.
- This is an opinion on a future direction.

## Reference

- Original post: [https://x.com/CFGeek/status/2057508157854007515](https://x.com/CFGeek/status/2057508157854007515)
- Author: [@CFGeek](https://x.com/CFGeek) on X, posted 2026-05-21
