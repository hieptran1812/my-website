---
title: "Contrastive Language Model as a Fast System One Model"
publishDate: "2026-09-24"
date: "2026-09-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "A reaction to CLM, a contrastively trained model linking states and actions, claimed to be up to 9x faster than Jev."
---

> Source: [@hxiao on X](https://x.com/hxiao/status/2102930096827118053) · posted 2026-09-24

## Summary

The quoted post introduces **Contrastive Language Model (CLM)**, a fast System One model trained with a contrastive objective connecting states and actions. CLM-8B is pretrained on internet-scale data and is claimed to be up to 9x faster than Jev at comparable quality. The author likes its use of InfoNCE and its handling of long-horizon context, but notes that actions are still encoded independently, so it is not a listwise architecture like jina-reranker-v3/3.5.

## Background

Contrastive learning, as in CLIP, trains a model to place matching pairs close together and non-matching pairs apart. InfoNCE is the standard loss for this. A listwise model looks at all candidates together, while independent encoding scores each candidate alone, which is faster but cannot compare options directly.

## What I took away

- Contrastive scoring of state-action pairs can be much faster than generating text.
- Independent encoding versus listwise scoring is a key design trade-off in rankers.
- Treat the 9x speed claim as the vendor's own number.

## Reference

- Original post: [https://x.com/hxiao/status/2102930096827118053](https://x.com/hxiao/status/2102930096827118053)
- Author: [@hxiao](https://x.com/hxiao) on X, posted 2026-09-24
