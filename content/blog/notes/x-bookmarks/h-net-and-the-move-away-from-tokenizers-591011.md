---
title: "H-Net and the Move Away From Tokenizers"
publishDate: "2026-10-29"
date: "2026-10-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A commentator sees an AI research renaissance as H-Net learns to dynamically chunk raw data instead of using BPE."
---

> Source: [@gordic_aleksa on X](https://x.com/gordic_aleksa/status/2115920694974591011) · posted 2026-10-29

## Summary

@gordic_aleksa says we are in an AI research renaissance and finally getting rid of tokenizers. Commenting on H-Net's earlier paper, he explains that instead of a fixed BPE tokenizer, the model learns to dynamically chunk raw data. He quotes the post arguing for general building blocks that learn their own abstractions.

## Background

Tokenizers split text into units before training and are fixed. Dynamic chunking learns which byte groups to treat as units, adapting to the data. This could help with code, other languages and new modalities.

## What I took away

- Tokenizer-free models are an active research direction.
- Learned chunking may handle diverse data better than fixed vocabularies.
- Read the original H-Net paper for the mechanism.

## Reference

- Original post: [https://x.com/gordic_aleksa/status/2115920694974591011](https://x.com/gordic_aleksa/status/2115920694974591011)
- Author: [@gordic_aleksa](https://x.com/gordic_aleksa) on X, posted 2026-10-29
