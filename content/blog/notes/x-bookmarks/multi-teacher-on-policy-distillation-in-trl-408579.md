---
title: "Multi-Teacher On-Policy Distillation In TRL"
publishDate: "2026-11-08"
date: "2026-11-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "A weekend reading pick about multi-teacher on-policy distillation, now supported in Hugging Face TRL."
---

> Source: [@SergioPaniego on X](https://x.com/SergioPaniego/status/2119452382694408579) · posted 2026-11-08

## Summary

The author recommends learning about multi-teacher on-policy distillation (MOPD), which many frontier open models use in post-training this year. It is now supported in TRL, and the blog includes an example runnable on HF Jobs with small models.

## Background

On-policy distillation trains a student on its own samples while teachers grade them. With several teachers, the student learns from each expert. TRL is Hugging Face's training library.

## What I took away

- MOPD merges several expert skills into one model.
- You can try it with small models.
- TRL support lowers the barrier.

## Reference

- Original post: [https://x.com/SergioPaniego/status/2119452382694408579](https://x.com/SergioPaniego/status/2119452382694408579)
- Author: [@SergioPaniego](https://x.com/SergioPaniego) on X, posted 2026-11-08
