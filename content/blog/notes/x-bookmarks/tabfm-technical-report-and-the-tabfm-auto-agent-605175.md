---
title: "TabFM Technical Report and the TabFM-Auto Agent"
publishDate: "2026-09-30"
date: "2026-09-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "ai-agents", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "The TabFM tabular foundation model gets a technical report, plus TabFM-Auto, an LLM agent that does feature engineering on top of it."
---

> Source: [@DeqingFu on X](https://x.com/DeqingFu/status/2105120581385605175) · posted 2026-09-30

## Summary

The quoted post introduces **TabFM**, a foundation model for tabular classification and regression that predicts on unseen tables in a single forward pass. The author now releases its technical report and introduces **TabFM-Auto**, an LLM agent that does feature engineering while TabFM stays frozen. Links to both papers and an explorer of the features the LLMs found are shared.

## Background

Tabular data means spreadsheet-like rows and columns, where tools like gradient-boosted trees usually win. A tabular foundation model is pretrained on many tables so it can predict on a new table without training from scratch. Feature engineering means creating new input columns from existing ones, a task that usually takes a lot of human effort.

## What I took away

- Tabular foundation models aim to replace per-dataset training with one forward pass.
- Pairing a frozen model with an LLM agent that edits the inputs is a cheap way to improve results without retraining.
- Inspecting the features an agent creates can teach you about your own data.

## Reference

- Original post: [https://x.com/DeqingFu/status/2105120581385605175](https://x.com/DeqingFu/status/2105120581385605175)
- Author: [@DeqingFu](https://x.com/DeqingFu) on X, posted 2026-09-30
