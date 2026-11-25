---
title: "Original Scaling Laws Had A Bug"
publishDate: "2026-07-04"
date: "2026-07-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A post notes the original scaling laws were wrong due to a bug, possibly wasting compute on oversized models."
---

> Source: [@sedielem on X](https://x.com/sedielem/status/2073446445307617366) · posted 2026-07-04

## Summary

@sedielem shares lore that the original scaling laws contained a bug, which probably led to wasted compute on oversized undertrained models, before inference costs were even counted.

## Background

Scaling laws predict how loss falls with model size and data. If they favor too-large models, labs train big models on too few tokens. Later work recommended more data per parameter.

## What I took away

- Scaling law fits can be wrong and have real cost.
- Count inference cost when sizing models.
- Check how a law was fit before relying on it.

## Reference

- Original post: [https://x.com/sedielem/status/2073446445307617366](https://x.com/sedielem/status/2073446445307617366)
- Author: [@sedielem](https://x.com/sedielem) on X, posted 2026-07-04
