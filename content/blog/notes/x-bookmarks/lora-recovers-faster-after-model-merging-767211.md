---
title: "LoRA Recovers Faster After Model Merging"
publishDate: "2025-12-14"
date: "2025-12-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2000076222135767211-1.jpg"
excerpt: "Follow-up observations that LoRA drops more after merging at first but recovers and improves faster in RL."
---

> Source: [@FanqingMengAI on X](https://x.com/FanqingMengAI/status/2000076222135767211) · posted 2025-12-14

![LoRA Recovers Faster After Model Merging](/imgs/blogs/x-bookmarks/2000076222135767211-1.jpg)

## Summary

@FanqingMengAI first thought the result meant LoRA forgets less, which would reduce loss from model merging. But they observed LoRA dropped more right after merging, yet recovered and improved faster during later training. The quoted post describes the plan to train experts per sub-task, merge them, then run multi-task RL.

## Background

Merging combines weights from several trained models. A model can lose performance right after, then regain it with more training. The observation suggests LoRA's benefit is about how quickly it recovers, not only about forgetting less.

## What I took away

- Judge a merging method by recovery after training, not just the immediate drop.
- LoRA may adapt faster after a merge.
- These are early experiments on ReasonGym.

## Reference

- Original post: [https://x.com/FanqingMengAI/status/2000076222135767211](https://x.com/FanqingMengAI/status/2000076222135767211)
- Author: [@FanqingMengAI](https://x.com/FanqingMengAI) on X, posted 2025-12-14
