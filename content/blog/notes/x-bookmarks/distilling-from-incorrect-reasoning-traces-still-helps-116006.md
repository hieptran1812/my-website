---
title: "Distilling From Incorrect Reasoning Traces Still Helps"
publishDate: "2026-01-10"
date: "2026-01-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2009995065243116006-1.png"
excerpt: "A finding that fine-tuning on chain-of-thought traces ending in wrong answers can still improve reasoning."
---

> Source: [@agarwl_ on X](https://x.com/agarwl_/status/2009995065243116006) · posted 2026-01-10

![Distilling From Incorrect Reasoning Traces Still Helps](/imgs/blogs/x-bookmarks/2009995065243116006-1.png)

## Summary

The author asks what happens if you distill a large LLM into a smaller one using only incorrect reasoning traces, judged by final answer correctness. They say it still works. The quoted paper thread says finetuning on synthetic chain-of-thought from a more capable model can improve reasoning even when every trace ends wrong.

## Background

Distillation trains a smaller model on outputs from a bigger one. A chain-of-thought trace is the step-by-step reasoning text. The surprise is that the reasoning steps carry useful signal even when the final answer is wrong.

## What I took away

- Final-answer correctness may be too strict a filter for training data.
- Reasoning steps can teach even when the answer is wrong.
- Read the paper to see the conditions where this holds.

## Reference

- Original post: [https://x.com/agarwl_/status/2009995065243116006](https://x.com/agarwl_/status/2009995065243116006)
- Author: [@agarwl_](https://x.com/agarwl_) on X, posted 2026-01-10
