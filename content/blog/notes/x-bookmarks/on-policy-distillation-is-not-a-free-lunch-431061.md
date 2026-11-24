---
title: "On-Policy Distillation Is Not a Free Lunch"
publishDate: "2026-07-21"
date: "2026-07-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "Bhavin Jawade discusses the hidden costs of on-policy distillation, now a default tool in many open frontier training recipes."
---

> Source: [@BhavinJawade on X](https://x.com/BhavinJawade/status/2079604860287431061) · posted 2026-07-21

## Summary

The author argues that **on-policy distillation** is not a free lunch. He notes it has become a default post-training tool in many open-source frontier model recipes. Recent releases such as DeepSeek v4, MiMO and Nemotron-Cascade-2 use MOPD heavily (text cut off).

## Background

Distillation trains a student model to match a teacher model. On-policy means the student generates its own answers and the teacher scores or corrects them, which avoids the mismatch of training only on teacher outputs. MOPD likely refers to a multi-teacher variant; the costs can include compute and inherited teacher limits.

## What I took away

- Popular techniques still have trade-offs worth understanding.
- Weigh the compute cost of running a teacher during training.
- Read the full thread for the specific failure modes.

## Reference

- Original post: [https://x.com/BhavinJawade/status/2079604860287431061](https://x.com/BhavinJawade/status/2079604860287431061)
- Author: [@BhavinJawade](https://x.com/BhavinJawade) on X, posted 2026-07-21
