---
title: "SAE Features Do Not Simply Add Up"
publishDate: "2026-10-18"
date: "2026-10-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A paper introduces SAE Additivity Residuals to measure behavior that individual sparse autoencoder features cannot explain."
---

> Source: [@HuggingPapers on X](https://x.com/HuggingPapers/status/2111614913969844503) · posted 2026-10-18

## Summary

@HuggingPapers shares a paper saying sparse autoencoder features do not simply add up. It introduces SAE Additivity Residuals (SAR), which quantify behavioral variation that individual features cannot explain. Across 18 settings, learned coordinates beat random rotations, and pairwise interaction terms recover much of what is missing.

## Background

A sparse autoencoder (SAE) decomposes a model's activations into many features, each ideally meaning one concept. Interpretability often assumes effects of features combine additively. If features interact, explaining behavior needs pairs or groups of features.

## What I took away

- Single-feature explanations can miss interaction effects.
- Measure how much behavior your features actually explain.
- This adds nuance to the neurons-versus-features debate in this batch.

## Reference

- Original post: [https://x.com/HuggingPapers/status/2111614913969844503](https://x.com/HuggingPapers/status/2111614913969844503)
- Author: [@HuggingPapers](https://x.com/HuggingPapers) on X, posted 2026-10-18
