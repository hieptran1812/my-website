---
title: "Internal Data Repetition Wastes Compute"
publishDate: "2026-07-05"
date: "2026-07-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2073562544069857610-1.jpg"
excerpt: "Stanford AI Lab shares work measuring how leftover duplicate data wastes up to 33% of compute."
---

> Source: [@StanfordAILab on X](https://x.com/StanfordAILab/status/2073562544069857610) · posted 2026-07-05

![Internal Data Repetition Wastes Compute](/imgs/blogs/x-bookmarks/2073562544069857610-1.jpg)

## Summary

@StanfordAILab says deduplication is never perfect, and this work measures the leftover cost in compute terms, showing worst-case repetition is predictable from model size, wasting as much as 33% of compute. The quoted post presents the paper Internal Data Repetition Destroys Language Models.

## Background

Deduplication removes repeated text from training data. Repetition inside a document can still remain. Since pretraining is data-constrained, repeats hurt more.

## What I took away

- Imperfect dedup has a measurable compute cost.
- Worst-case repetition depends on model size.
- Check your data for internal repeats, not only duplicates across documents.

## Reference

- Original post: [https://x.com/StanfordAILab/status/2073562544069857610](https://x.com/StanfordAILab/status/2073562544069857610)
- Author: [@StanfordAILab](https://x.com/StanfordAILab) on X, posted 2026-07-05
