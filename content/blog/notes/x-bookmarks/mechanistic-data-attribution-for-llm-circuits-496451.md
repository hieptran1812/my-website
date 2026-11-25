---
title: "Mechanistic Data Attribution for LLM Circuits"
publishDate: "2026-02-03"
date: "2026-02-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2018625817115496451-1.jpg"
excerpt: "Introduces Mechanistic Data Attribution, which uses influence functions to trace interpretable LLM circuits back to training data."
---

> Source: [@PanLiangming on X](https://x.com/PanLiangming/status/2018625817115496451) · posted 2026-02-03

![Mechanistic Data Attribution for LLM Circuits](/imgs/blogs/x-bookmarks/2018625817115496451-1.jpg)

## Summary

The post says mechanistic interpretability has found interpretable circuits in LLMs, but where they come from in the training data is unknown. The authors introduce Mechanistic Data Attribution (MDA), a scalable framework that uses influence functions to bridge this gap.

## Background

A circuit is a small group of model components that together perform a specific behavior. Influence functions estimate how much each training example contributed to a model's behavior. Linking circuits to the data that created them could explain why models learn certain skills.

## What I took away

- Interpretability is moving from what a model does to why it learned it.
- Influence functions are a tool for tracing behavior to training examples.
- Look at the paper to see how the method scales.

## Reference

- Original post: [https://x.com/PanLiangming/status/2018625817115496451](https://x.com/PanLiangming/status/2018625817115496451)
- Author: [@PanLiangming](https://x.com/PanLiangming) on X, posted 2026-02-03
