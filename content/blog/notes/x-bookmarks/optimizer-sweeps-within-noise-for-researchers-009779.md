---
title: "Optimizer Sweeps Within Noise For Researchers"
publishDate: "2026-05-09"
date: "2026-05-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2052954169989009779-1.jpg"
excerpt: "A post asks whether a quoted optimizer comparison is a northern lights moment, where results fall within noise."
---

> Source: [@samsja19 on X](https://x.com/samsja19/status/2052954169989009779) · posted 2026-05-09

![Optimizer Sweeps Within Noise For Researchers (1)](/imgs/blogs/x-bookmarks/2052954169989009779-1.jpg)

![Optimizer Sweeps Within Noise For Researchers (2)](/imgs/blogs/x-bookmarks/2052954169989009779-2.jpg)

## Summary

The author asks if this is northern lights for AI researchers. The quoted post says earlier 20-trial 3200-step sweeps were undersampled. At 100 trials with 800 steps everything seems within noise, with ranking SOAP > Muon+AdamW > KLSOAP > AdamW, which contradicts what the math predicts.

## Background

SOAP, Muon and AdamW are optimizers used to train neural networks. A sweep tries many hyperparameter settings. If differences are within noise, small experiments can mislead, so more trials are needed.

## What I took away

- Small sweeps can rank optimizers wrongly; use enough trials.
- When results contradict theory, check the experiment before the math.

## Reference

- Original post: [https://x.com/samsja19/status/2052954169989009779](https://x.com/samsja19/status/2052954169989009779)
- Author: [@samsja19](https://x.com/samsja19) on X, posted 2026-05-09
