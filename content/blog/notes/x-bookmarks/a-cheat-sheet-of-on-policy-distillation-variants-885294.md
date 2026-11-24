---
title: "A Cheat Sheet of On-Policy Distillation Variants"
publishDate: "2026-07-19"
date: "2026-07-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
excerpt: "A compact list of distillation objectives such as OPD, OPSD, RLSD and OPD squared, with one formula each."
---

> Source: [@ethantsliu on X](https://x.com/ethantsliu/status/2078672504298885294) · posted 2026-07-19

## Summary

The author lists several distillation techniques with short formulas: OPD (teacher log-prob minus student log-prob), OPSD (self-distillation conditioned on a reference answer), RLSD (advantage direction with OPSD magnitude), and OPD squared. The quoted post explains OPD squared: normal OPD copies everything the teacher prefers, including habits from before reasoning tuning, while OPD squared uses the token-level log-prob change between the reasoning teacher and its own base checkpoint, so the reward isolates what post-training added.

## Background

On-policy distillation means the student generates its own text and the teacher scores each token, which avoids the mismatch of training only on teacher outputs. Log-prob differences act as a per-token reward. Subtracting the base model's preferences is a way to keep only the skill the teacher gained from post-training.

## What I took away

- Many distillation methods differ only in what log-prob difference they use as the signal.
- Subtracting a base checkpoint can filter out unwanted inherited habits.
- Keep this list as a reference when reading post-training papers.

## Reference

- Original post: [https://x.com/ethantsliu/status/2078672504298885294](https://x.com/ethantsliu/status/2078672504298885294)
- Author: [@ethantsliu](https://x.com/ethantsliu) on X, posted 2026-07-19
