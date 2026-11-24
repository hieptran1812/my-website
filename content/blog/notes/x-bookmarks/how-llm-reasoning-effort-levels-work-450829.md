---
title: "How LLM Reasoning Effort Levels Work"
publishDate: "2026-07-18"
date: "2026-07-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "post-training", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Sebastian Raschka wrote an article on how low, medium and high reasoning effort is implemented at inference and during training."
---

> Source: [@rasbt on X](https://x.com/rasbt/status/2078471977237450829) · posted 2026-07-18

## Summary

The author asks how an LLM switches between low, medium and high reasoning effort, and how it learns to reason more or less. He says he wrote an article explaining how these effort levels are implemented at inference time and during training.

## Background

Reasoning models produce a chain of thought before answering. Effort levels control how long that thinking is, which trades accuracy for cost and latency. This can be done with prompts, token budgets at inference time, or by training the model on examples of different lengths.

## What I took away

- Effort control is both an inference-time and a training-time design choice.
- Read the article to understand the trade-off between thinking length and cost.
- When using APIs with effort settings, measure whether higher effort actually helps your task.

## Reference

- Original post: [https://x.com/rasbt/status/2078471977237450829](https://x.com/rasbt/status/2078471977237450829)
- Author: [@rasbt](https://x.com/rasbt) on X, posted 2026-07-18
