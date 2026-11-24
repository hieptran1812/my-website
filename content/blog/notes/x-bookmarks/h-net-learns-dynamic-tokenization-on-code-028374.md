---
title: "H-Net Learns Dynamic Tokenization on Code"
publishDate: "2026-10-28"
date: "2026-10-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "pretraining"]
author: "Hiep Tran"
featured: false
excerpt: "H-Net learns to tokenize variable names as separate units and scales better on code than BPE."
---

> Source: [@tri_dao on X](https://x.com/tri_dao/status/2115531348161028374) · posted 2026-10-28

## Summary

The quoted post argues intelligence should come from simple, general building blocks that learn their own abstractions from raw data, the bitter lesson, and sees a real step toward AI that learns from all data. @tri_dao highlights a result: H-Net learns to dynamically tokenize variable names in code as separate tokens and scales much better on code than BPE.

## Background

BPE (byte pair encoding) is a fixed tokenizer that splits text into common chunks before training. H-Net instead learns how to chunk raw bytes during training. The bitter lesson is the idea that general methods using more compute beat hand-designed ones.

## What I took away

- Fixed tokenizers may limit models on domains like code.
- Learned chunking lets the model find meaningful units itself.
- Watch tokenizer-free models as a possible next architecture shift.

## Reference

- Original post: [https://x.com/tri_dao/status/2115531348161028374](https://x.com/tri_dao/status/2115531348161028374)
- Author: [@tri_dao](https://x.com/tri_dao) on X, posted 2026-10-28
