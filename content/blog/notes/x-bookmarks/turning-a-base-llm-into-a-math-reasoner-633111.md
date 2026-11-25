---
title: "Turning A Base LLM Into A Math Reasoner"
publishDate: "2025-12-06"
date: "2025-12-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "reinforcement-learning", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "An interview-style post about how to turn a weak base LLM into a strong maths and reasoning model, hinting that plain fine-tuning is not enough."
---

> Source: [@akshay_pachaar on X](https://x.com/akshay_pachaar/status/1997284196306633111) · posted 2025-12-06

## Summary

@akshay_pachaar sets up a Google research scientist interview: a base LLM is bad at maths, and you are asked how to make it a maths and reasoning powerhouse. The wrong answer is to label some problems and fine-tune, and the thread explains what was missed.

## Background

Supervised fine-tuning teaches a model from labeled examples, which is limited by how much labeled data you have. Reinforcement learning with verifiable rewards lets a model try many solutions and learn from whether the final answer is right. This is how many recent reasoning models are trained.

## What I took away

- Plain supervised fine-tuning on labeled problems is rarely the full answer.
- Verifiable rewards allow models to learn reasoning from their own attempts.

## Reference

- Original post: [https://x.com/akshay_pachaar/status/1997284196306633111](https://x.com/akshay_pachaar/status/1997284196306633111)
- Author: [@akshay_pachaar](https://x.com/akshay_pachaar) on X, posted 2025-12-06
