---
title: "NEPA Next-Embedding Predictive Autoregression Announced"
publishDate: "2025-12-19"
date: "2025-12-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "Authors present NEPA, an autoregressive model that predicts the next embedding for visual self-supervised learning."
---

> Source: [@ziqiao_ma on X](https://x.com/ziqiao_ma/status/2002096476437295392) · posted 2025-12-19

## Summary

The post announces NEPA, Next-Embedding Predictive Autoregression, a simple objective for visual self-supervised learning and generative pretraining. Instead of reconstructing pixels or predicting discrete tokens, an autoregressive model predicts the next embedding given all previous ones. The key ideas are listed after the cut-off.

## Background

Autoregressive means predicting the next item from all earlier items, as language models do with tokens. Here the items are image embeddings, avoiding both pixel reconstruction and discrete tokenizers.

## What I took away

- Next-embedding prediction removes the need for a visual tokenizer.
- It unifies representation learning and generation in one objective.

## Reference

- Original post: [https://x.com/ziqiao_ma/status/2002096476437295392](https://x.com/ziqiao_ma/status/2002096476437295392)
- Author: [@ziqiao_ma](https://x.com/ziqiao_ma) on X, posted 2025-12-19
