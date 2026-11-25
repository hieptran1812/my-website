---
title: "Elo And XGBoost Predicting Tennis Winners"
publishDate: "2026-03-10"
date: "2026-03-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quant", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A student used chess Elo ratings and XGBoost on 95,491 tennis matches to predict winners with 85% accuracy."
---

> Source: [@dunik_7 on X](https://x.com/dunik_7/status/2031452419717210322) · posted 2026-03-10

## Summary

The post says a student applied the Elo rating system to 95,491 tennis matches over 43 years and trained an XGBoost model that predicts winners with 85% accuracy. On the 2025 Australian Open, outside the training data, it got 99 of 116 matches right. These are the post's claims.

## Background

Elo gives each player a rating that updates after matches. XGBoost is a gradient boosted tree model. Using Elo as input features and testing on a later event is a sound way to check generalization.

## What I took away

- Ratings like Elo make strong features for tabular models.
- Test on a held-out later period, as done here.
- Verify the numbers independently.

## Reference

- Original post: [https://x.com/dunik_7/status/2031452419717210322](https://x.com/dunik_7/status/2031452419717210322)
- Author: [@dunik_7](https://x.com/dunik_7) on X, posted 2026-03-10
