---
title: "The Residual Connection Equation Behind Deep Learning"
publishDate: "2025-11-30"
date: "2025-11-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources", "math"]
author: "Hiep Tran"
featured: false
excerpt: "A post highlights the residual update h(k+1) = h(k) + f(h(k)) as the key to training deep networks."
---

> Source: [@chastronomic on X](https://x.com/chastronomic/status/1995111981473538351) · posted 2025-11-30

## Summary

The author says ML engineers use one equation daily: h(k+1) = h(k) + f(h(k)). They claim the small plus h(k) term is why deep networks scale and train. It is labelled Concept 21 in their series.

## Background

This is a residual (skip) connection. Each layer adds a change to its input instead of replacing it, so gradients flow back easily through many layers. It is used in ResNets and Transformers.

## What I took away

- Residual connections let you stack many layers.
- Think of each layer as a small correction to the input.

## Reference

- Original post: [https://x.com/chastronomic/status/1995111981473538351](https://x.com/chastronomic/status/1995111981473538351)
- Author: [@chastronomic](https://x.com/chastronomic) on X, posted 2025-11-30
