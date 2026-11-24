---
title: "Connection to Nearest Neighbor Language Models"
publishDate: "2026-10-28"
date: "2026-10-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A reply notes a result reminds her of kNN language models, which mix a neural LM with nearest-neighbor retrieval."
---

> Source: [@ZoeyLi20 on X](https://x.com/ZoeyLi20/status/2115556925152829594) · posted 2026-10-28

## Summary

@ZoeyLi20 says something reminds her of KNN language models and links the paper Generalization through Memorization: Nearest Neighbor Language Models. That paper extends a pretrained LM by interpolating it with a k-nearest-neighbors model.

## Background

A kNN-LM stores hidden states and next tokens from training text. At inference it finds the most similar stored contexts and blends their next tokens with the model's own prediction. This helps with rare facts the model cannot memorize.

## What I took away

- Retrieval can complement memorization in language models.
- Old ideas like kNN-LM often resurface in new results.
- Consider retrieval when a model struggles with rare content.

## Reference

- Original post: [https://x.com/ZoeyLi20/status/2115556925152829594](https://x.com/ZoeyLi20/status/2115556925152829594)
- Author: [@ZoeyLi20](https://x.com/ZoeyLi20) on X, posted 2026-10-28
