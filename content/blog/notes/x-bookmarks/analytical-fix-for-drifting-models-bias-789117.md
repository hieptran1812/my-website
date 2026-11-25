---
title: "Analytical Fix For Drifting Models Bias"
publishDate: "2026-11-06"
date: "2026-11-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research", "math"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2118740517827789117-1.jpg"
excerpt: "A NeurIPS 2026 paper corrects minibatch bias in drifting models with two extra lines of code."
---

> Source: [@ruqi_zhang on X](https://x.com/ruqi_zhang/status/2118740517827789117) · posted 2026-11-06

![Analytical Fix For Drifting Models Bias](/imgs/blogs/x-bookmarks/2118740517827789117-1.jpg)

## Summary

The author highlights Jiaru's work on debiasing drifting models. The quoted paper says minibatch drift is biased and ABC fixes it with 2 extra lines of code, reducing bias from O(1/n) to O(1/n²), with lower FID and faster training.

## Background

Drifting models are generative models trained with minibatch estimates. Subsampling creates bias. FID is an image quality score where lower is better. Claims are from the authors.

## What I took away

- Small code changes can reduce bias.
- Check minibatch estimator bias.
- Code is available.

## Reference

- Original post: [https://x.com/ruqi_zhang/status/2118740517827789117](https://x.com/ruqi_zhang/status/2118740517827789117)
- Author: [@ruqi_zhang](https://x.com/ruqi_zhang) on X, posted 2026-11-06
