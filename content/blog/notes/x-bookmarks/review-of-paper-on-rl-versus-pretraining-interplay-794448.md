---
title: "Review Of Paper On RL Versus Pretraining Interplay"
publishDate: "2025-12-16"
date: "2025-12-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001009781960794448-1.jpg"
excerpt: "Dhruv Batra praises a paper that studies when RL improves base model performance, using synthetic GPT-2 scale experiments."
---

> Source: [@DhruvBatra_ on X](https://x.com/DhruvBatra_/status/2001009781960794448) · posted 2025-12-16

![Review Of Paper On RL Versus Pretraining Interplay](/imgs/blogs/x-bookmarks/2001009781960794448-1.jpg)

## Summary

Dhruv Batra says he finished a really well done paper and notes meta-points, such as using GSM-inf as a test-bed for systematic experiments and a structure like a Science or Nature paper. The quoted post says views differ on whether RL really improves a base model (for example pass@128), and the answer is both yes and no, depending on the interplay of pre-training, mid-training and RL. They trained a few hundred GPT-2 scale models on synthetic data.

## Background

Pass@128 asks if any of 128 samples is correct, so it measures what the model can do at all, not only its first guess. Using small models on synthetic data lets researchers control exactly what was seen in pre-training. This helps separate real new skills from sharpening of existing ones.

## What I took away

- Synthetic controlled setups can answer questions that real-data studies cannot.
- Whether RL adds new ability depends on what the base model already knows.
- Read the paper for the conditions where RL helps.

## Reference

- Original post: [https://x.com/DhruvBatra_/status/2001009781960794448](https://x.com/DhruvBatra_/status/2001009781960794448)
- Author: [@DhruvBatra_](https://x.com/DhruvBatra_) on X, posted 2025-12-16
