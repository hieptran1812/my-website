---
title: "dLLMs Get On-Policy Self-Distillation With d-OPSD"
publishDate: "2026-06-20"
date: "2026-06-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "distillation", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2068316539976368539-1.png"
excerpt: "A post reacts to d-OPSD, an on-policy self-distillation framework for diffusion LLMs that beats RLVR with fewer steps."
---

> Source: [@sheriyuo on X](https://x.com/sheriyuo/status/2068316539976368539) · posted 2026-06-20

![dLLMs Get On-Policy Self-Distillation With d-OPSD (1)](/imgs/blogs/x-bookmarks/2068316539976368539-1.png)

![dLLMs Get On-Policy Self-Distillation With d-OPSD (2)](/imgs/blogs/x-bookmarks/2068316539976368539-2.jpg)

## Summary

@sheriyuo says dLLMs now have OPD too, and notes an idea they had was already a paper (arXiv 2605.26606), saying research is often a race. The quoted post introduces d-OPSD, learning from self-generated suffixes with step-level supervision, beating RLVR with about 10% of training steps.

## Background

Diffusion LLMs generate text by iterative refinement rather than left to right. Self-distillation uses the model as its own teacher. RLVR is RL with verifiable rewards. The 10% claim is the authors'.

## What I took away

- On-policy distillation now extends to diffusion LLMs.
- Ideas get scooped quickly.

## Reference

- Original post: [https://x.com/sheriyuo/status/2068316539976368539](https://x.com/sheriyuo/status/2068316539976368539)
- Author: [@sheriyuo](https://x.com/sheriyuo) on X, posted 2026-06-20
