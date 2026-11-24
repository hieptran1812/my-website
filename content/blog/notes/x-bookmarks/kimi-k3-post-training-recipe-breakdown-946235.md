---
title: "Kimi K3 Post-Training Recipe Breakdown"
publishDate: "2026-07-28"
date: "2026-07-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "moe", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Bhavin Jawade summarizes the post-training recipe of Kimi K3, a 2.8T-parameter open-weight MoE model, from its technical report."
---

> Source: [@BhavinJawade on X](https://x.com/BhavinJawade/status/2082134026475946235) · posted 2026-07-28

## Summary

The author says the **Kimi K3 technical report** has been released. K3 is a 2.8T-parameter open-weight MoE model. He went through the report and shares its post-training recipe, comparing it to the DeepSeek V4 recipe (text cut off).

## Background

MoE (mixture of experts) models route each token to a few expert sub-networks, so they can be very large but cheap per token. The post-training recipe is the sequence of fine-tuning, RL and distillation steps applied after pretraining. Comparing recipes across labs shows which methods are becoming standard.

## What I took away

- Post-training recipes are converging across open frontier labs.
- Read thread breakdowns as a guide, then check the original report.
- Note which steps K3 shares with DeepSeek V4.

## Reference

- Original post: [https://x.com/BhavinJawade/status/2082134026475946235](https://x.com/BhavinJawade/status/2082134026475946235)
- Author: [@BhavinJawade](https://x.com/BhavinJawade) on X, posted 2026-07-28
