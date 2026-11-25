---
title: "Quark: 165M Model Post-Trained With RL Only"
publishDate: "2026-11-15"
date: "2026-11-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "open-models", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2122019918686372154-1.jpg"
excerpt: "A post shares Quark, a 165M-parameter language model with 128K context trained with RL only, costing about $4,200."
---

> Source: [@mkurman88 on X](https://x.com/mkurman88/status/2122019918686372154) · posted 2026-11-15

![Quark: 165M Model Post-Trained With RL Only](/imgs/blogs/x-bookmarks/2122019918686372154-1.jpg)

## Summary

The author says watching Quark training was a pleasure. The quoted post describes a 165M-parameter model with 128K context, post-trained with RL only, with no SFT stage and no distillation. The whole project, including a failed first design, cost about $4,200 or 1,300 H100-hours, and released four open-weight models and a report.

## Background

Most pipelines use SFT or distillation before RL. Skipping them shows what RL alone can do for a small model. The numbers come from the quoted post.

## What I took away

- Small RL-only projects can be affordable.
- The report includes a failed design.
- Open weights allow reproduction.

## Reference

- Original post: [https://x.com/mkurman88/status/2122019918686372154](https://x.com/mkurman88/status/2122019918686372154)
- Author: [@mkurman88](https://x.com/mkurman88) on X, posted 2026-11-15
