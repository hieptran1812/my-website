---
title: "Predictive Data Debugging Before Training"
publishDate: "2026-06-11"
date: "2026-06-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2065120344185409740-1.jpg"
excerpt: "A post on finding problems in training data, such as broken guardrails, before a model learns them."
---

> Source: [@EkdeepL on X](https://x.com/EkdeepL/status/2065120344185409740) · posted 2026-06-11

![Predictive Data Debugging Before Training](/imgs/blogs/x-bookmarks/2065120344185409740-1.jpg)

## Summary

@EkdeepL says the work follows his claim that interpretability is the language of data. The quoted post introduces predictive data debugging, which reveals what a model will learn before training. In DPO datasets they found broken guardrails, hallucinations and fish fart fan fiction.

## Background

DPO trains a model from preferred versus rejected answers. Bad examples in such data teach bad behavior. Debugging data before training is cheaper than fixing the model afterward.

## What I took away

- Audit preference data before using it.
- Data bugs become model bugs.

## Reference

- Original post: [https://x.com/EkdeepL/status/2065120344185409740](https://x.com/EkdeepL/status/2065120344185409740)
- Author: [@EkdeepL](https://x.com/EkdeepL) on X, posted 2026-06-11
