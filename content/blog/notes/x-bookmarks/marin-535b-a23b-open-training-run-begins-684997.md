---
title: "Marin 535B-A23B Open Training Run Begins"
publishDate: "2026-08-21"
date: "2026-08-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "open-models", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "Percy Liang announced that the Marin 535B-A23B model has started training, with the whole process open."
---

> Source: [@percyliang on X](https://x.com/percyliang/status/2090918065634684997) · posted 2026-08-21

## Summary

Percy Liang says Marin 535B-A23B started training this week, with the process done in the open as usual. The plan is pretraining (80%) and midtraining (20%) on 18.75T tokens using 11 GB200 NVL72 systems for about three months, around 2.7e24 FLOPs, with post-training to follow. Before the main run, the team trained a smaller ladder of models.

## Background

A name like 535B-A23B usually means a mixture-of-experts model with 535 billion total parameters, of which about 23 billion are active per token. Midtraining is a phase between pretraining and post-training that often uses higher-quality or specialized data. Teams commonly train a ladder of smaller models first to check that settings scale before spending compute on the big run.

## What I took away

- Open training runs are a rare chance to see real compute budgets and plans.
- Training a ladder of small models first is a standard way to reduce risk on a large run.
- Watch for the post-training phase and how results compare to the plan.

## Reference

- Original post: [https://x.com/percyliang/status/2090918065634684997](https://x.com/percyliang/status/2090918065634684997)
- Author: [@percyliang](https://x.com/percyliang) on X, posted 2026-08-21
