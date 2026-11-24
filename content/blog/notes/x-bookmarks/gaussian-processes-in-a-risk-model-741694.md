---
title: "Gaussian Processes in a Risk Model"
publishDate: "2026-11-01"
date: "2026-11-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quant", "math", "trading"]
author: "Hiep Tran"
featured: false
excerpt: "A trader says Gaussian processes are part of his current risk model, quoting an intro to GPs."
---

> Source: [@mmmatt on X](https://x.com/mmmatt/status/2116956598748741694) · posted 2026-11-01

## Summary

The quoted post introduces Gaussian Processes (GPs) as a framework for Bayesian nonparametric regression and classification that places a probability distribution directly over functions, defined by a mean function m and a kernel k. @mmmatt says GPs are part of his current risk model and that they are very powerful.

## Background

A Gaussian process predicts a whole distribution over possible functions, so it gives both a prediction and an uncertainty. The kernel defines how similar two inputs are and therefore how smooth the function is. Uncertainty estimates are useful in risk modeling.

## What I took away

- GPs give uncertainty along with predictions, which suits risk work.
- Kernel choice controls the assumptions a GP makes.
- GPs scale poorly to very large datasets without approximations.

## Reference

- Original post: [https://x.com/mmmatt/status/2116956598748741694](https://x.com/mmmatt/status/2116956598748741694)
- Author: [@mmmatt](https://x.com/mmmatt) on X, posted 2026-11-01
