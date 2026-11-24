---
title: "DeepLoop Makes Looped Transformers Stable and Scalable"
publishDate: "2026-09-03"
date: "2026-09-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A paper called DeepLoop proposes depth scaling for looped transformers to make them stable and scalable."
---

> Source: [@yacinelearning on X](https://x.com/yacinelearning/status/2095459251099517313) · posted 2026-09-03

## Summary

The author shows strong support for the researcher behind the quoted work. The quoted post says it has heard that some frontier models are basically a 48-layer transformer looped twice, and introduces **DeepLoop: Depth Scaling for Looped Transformers**, a paper aimed at making looped transformers stable and scalable.

## Background

A looped transformer runs the same layers more than once, so a 48-layer model looped twice acts like 96 layers of compute with 48 layers of weights. Training such models can be unstable because errors and activations compound through repeated passes, so methods that scale depth safely matter.

## What I took away

- Looping layers trades extra compute for fewer parameters.
- Stability is the main practical obstacle for looped transformers.
- The claim about frontier models being looped is hearsay in the post, not a confirmed fact.

## Reference

- Original post: [https://x.com/yacinelearning/status/2095459251099517313](https://x.com/yacinelearning/status/2095459251099517313)
- Author: [@yacinelearning](https://x.com/yacinelearning) on X, posted 2026-09-03
