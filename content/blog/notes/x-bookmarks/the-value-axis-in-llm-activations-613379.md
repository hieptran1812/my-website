---
title: "The Value Axis In LLM Activations"
publishDate: "2026-06-17"
date: "2026-06-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2067298443316613379-1.jpg"
excerpt: "A paper finds LLMs track the chance of reaching their goal along a linear axis, like an RL value function."
---

> Source: [@Jack_W_Lindsey on X](https://x.com/Jack_W_Lindsey/status/2067298443316613379) · posted 2026-06-17

![The Value Axis In LLM Activations](/imgs/blogs/x-bookmarks/2067298443316613379-1.jpg)

## Summary

@Jack_W_Lindsey says LLMs encode whether they are on the right track along a linear axis, like a value function. The axis shifts when training new preferences and modulates confidence and the likelihood of backtracking. The quoted post says it affects math and coding and can be reshaped with DPO and SFT.

## Background

A value function in RL estimates expected success. A linear axis means one direction in activation space. Finding it suggests models internally estimate progress.

## What I took away

- Models may estimate their own progress internally.
- The axis can be reshaped by training.

## Reference

- Original post: [https://x.com/Jack_W_Lindsey/status/2067298443316613379](https://x.com/Jack_W_Lindsey/status/2067298443316613379)
- Author: [@Jack_W_Lindsey](https://x.com/Jack_W_Lindsey) on X, posted 2026-06-17
