---
title: "Open Guide to Training Mixture-of-Experts Models"
publishDate: "2026-11-05"
date: "2026-11-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "pretraining", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A team spent 800k GPU hours on MoE training experiments and wrote everything down, arguing dense models are basically over."
---

> Source: [@TheZachMueller on X](https://x.com/TheZachMueller/status/2118456452306088252) · posted 2026-11-05

## Summary

The quoted post says 18 of the 20 best open models today are Mixture-of-Experts, that dense is basically over, and that MoE training know-how lives mostly within a few teams. So they spent 800k GPU hours running experiments and documented the results. @TheZachMueller praises it as another strong release from the @Thom_Wolf team.

## Background

A Mixture-of-Experts (MoE) model has many expert sub-networks and a router that sends each token to a few of them. This gives large capacity at lower compute per token. Training them well involves tricky choices like router design and load balancing, which is why a public write-up of experiments is valuable.

## What I took away

- MoE is now the dominant design among top open models, per the post.
- Public large-scale ablation write-ups save others huge compute costs.
- Read the guide for practical MoE choices before training your own.

## Reference

- Original post: [https://x.com/TheZachMueller/status/2118456452306088252](https://x.com/TheZachMueller/status/2118456452306088252)
- Author: [@TheZachMueller](https://x.com/TheZachMueller) on X, posted 2026-11-05
