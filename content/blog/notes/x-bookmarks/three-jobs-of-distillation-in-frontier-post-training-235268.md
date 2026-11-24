---
title: "Three Jobs of Distillation in Frontier Post-Training"
publishDate: "2026-07-08"
date: "2026-07-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A write-up on how frontier models use distillation in post-training: compression, merging RL experts and self-teaching."
---

> Source: [@SergioPaniego on X](https://x.com/SergioPaniego/status/2074878675439235268) · posted 2026-07-08

## Summary

The author says frontier models use distillation as a step in post-training. In 2026 he sees three jobs: compressing a big model into a small one, merging RL experts into a single model, and letting a model teach itself. He wrote up which models use each approach and how.

## Background

Distillation trains one model to match another's outputs. Merging RL experts means training separate specialist models with reinforcement learning, then distilling them into one generalist. Self-distillation uses a model's own better outputs, for example with more context or thinking, as the teacher.

## What I took away

- Distillation is now a core post-training tool, not just for compression.
- Specialist-then-merge is a common recipe worth understanding.
- Read the write-up for which models use which approach.

## Reference

- Original post: [https://x.com/SergioPaniego/status/2074878675439235268](https://x.com/SergioPaniego/status/2074878675439235268)
- Author: [@SergioPaniego](https://x.com/SergioPaniego) on X, posted 2026-07-08
