---
title: "DroPE Drops Positional Embeddings To Extend Context"
publishDate: "2026-01-12"
date: "2026-01-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2010565943269999091-1.jpg"
excerpt: "Sakana's DroPE removes positional embeddings after pretraining and recalibrates to unlock long context."
---

> Source: [@hardmaru on X](https://x.com/hardmaru/status/2010565943269999091) · posted 2026-01-12

![DroPE Drops Positional Embeddings To Extend Context](/imgs/blogs/x-bookmarks/2010565943269999091-1.jpg)

## Summary

hardmaru says positional embeddings are training wheels: they help convergence but hurt long-context generalization. Deleting them after pretraining and recalibrating with under 1% of the original budget unlocks massive context windows. The quoted post introduces DroPE for extending pretrained LLMs.

## Background

Positional embeddings tell the model token order. Models tied to trained lengths struggle beyond them.

## What I took away

- Position encoding may be needed only during training.
- Cheap recalibration can extend context.

## Reference

- Original post: [https://x.com/hardmaru/status/2010565943269999091](https://x.com/hardmaru/status/2010565943269999091)
- Author: [@hardmaru](https://x.com/hardmaru) on X, posted 2026-01-12
