---
title: "Unified Training Across All Foundation Model Phases"
publishDate: "2026-10-30"
date: "2026-10-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "post-training", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
excerpt: "Researchers propose a single algorithm to replace separate pretraining, midtraining and post-training stages."
---

> Source: [@Zanette_ai on X](https://x.com/Zanette_ai/status/2116201481686671704) · posted 2026-10-30

## Summary

@Zanette_ai asks whether all phases of foundation model training can be unified into one algorithm and says they have a compelling answer. The quoted post introduces unified training, a single algorithm for the full pipeline instead of separate pretraining, midtraining and post-training. They say it gives simpler infrastructure and a framework for better science.

## Background

Models are usually trained in stages: pretraining on raw text, midtraining on curated data, then post-training with SFT or RL. Each stage needs its own code and tuning. A single algorithm could make the pipeline simpler and easier to study.

## What I took away

- Training stages may be convention rather than necessity.
- Simpler infrastructure is a practical benefit of unification.
- Read the paper to see how they handle the shift from text to rewards.

## Reference

- Original post: [https://x.com/Zanette_ai/status/2116201481686671704](https://x.com/Zanette_ai/status/2116201481686671704)
- Author: [@Zanette_ai](https://x.com/Zanette_ai) on X, posted 2026-10-30
