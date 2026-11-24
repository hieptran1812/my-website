---
title: "Removing Whole Layers From Qwen3-8B With LaCo"
publishDate: "2026-10-19"
date: "2026-10-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "quantization", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A post claims layer pruning could shrink huge models for on-device AI, reacting to removing up to 10 layers from Qwen3-8B."
---

> Source: [@arnavmehta007 on X](https://x.com/arnavmehta007/status/2112096817320190403) · posted 2026-10-19

## Summary

@arnavmehta007 argues this could shrink 1T+ parameter models to a few hundred billion without losing much score, and combined with quantization and ternary ops could make on-device AI much faster. The quoted post says they removed up to 10 layers from Qwen3-8B-Base with LaCo (Layer Collapse); deleting 6 layers made it 16.7% smaller while keeping 78 to 94% of reasoning scores at a 1.68x perplexity ratio, with code and weights released.

## Background

Layer pruning removes entire transformer layers, often merging their weights into neighbors, to make a model smaller and faster. Perplexity measures how well a model predicts text; higher is worse. Losing up to 22% of reasoning scores is a real cost, so the extrapolation to trillion-parameter models is the author's speculation.

## What I took away

- Layer pruning saves size but can cost noticeable quality.
- Do not extrapolate small-model pruning results to much larger models without evidence.
- Combine pruning with light retraining to recover quality.

## Reference

- Original post: [https://x.com/arnavmehta007/status/2112096817320190403](https://x.com/arnavmehta007/status/2112096817320190403)
- Author: [@arnavmehta007](https://x.com/arnavmehta007) on X, posted 2026-10-19
