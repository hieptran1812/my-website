---
title: "Overfitting Concerns in a Momentum Factor Paper"
publishDate: "2026-08-04"
date: "2026-08-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quant", "trading", "math"]
author: "Hiep Tran"
featured: false
excerpt: "A quant critique pointing out that a factor paper reuses one sample for construction, testing and regressions, risking overfitting."
---

> Source: [@quantbeckman on X](https://x.com/quantbeckman/status/2084565603348525162) · posted 2026-08-04

## Summary

The author criticizes a factor paper whose own authors report low typical individual-factor Sharpe and admit possible overfitting. The same 1963 to 2014 sample is used to build the factors, test momentum over many lag and lookback choices, and estimate spanning regressions. The quoted post, by the same author, argues that most real portfolios are off the capital market line, so many combinations of beliefs, constraints and preferences can explain the same observed point.

## Background

A Sharpe ratio measures return per unit of risk. Using the same data to design and test a strategy inflates results because you pick what happened to work. The capital market line (CML) is the set of optimal risk-return portfolios in classic portfolio theory.

## What I took away

- Keep separate data for designing and testing a strategy.
- Testing many parameter combinations raises the chance of false discoveries.
- Educational critique only, not a trading recommendation.

## Reference

- Original post: [https://x.com/quantbeckman/status/2084565603348525162](https://x.com/quantbeckman/status/2084565603348525162)
- Author: [@quantbeckman](https://x.com/quantbeckman) on X, posted 2026-08-04
