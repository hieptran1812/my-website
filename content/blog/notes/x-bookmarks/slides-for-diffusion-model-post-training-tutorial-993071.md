---
title: "Slides for Diffusion Model Post-Training Tutorial"
publishDate: "2026-10-08"
date: "2026-10-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "post-training", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Slides from an ECCV 2026 tutorial on aligning diffusion models with reward gradients, DPO, GRPO and forward-process RL."
---

> Source: [@LiangZheng_06 on X](https://x.com/LiangZheng_06/status/2107986518312993071) · posted 2026-10-08

## Summary

The authors released slides from their ECCV'26 tutorial on diffusion model post-training. Topics include direct reward gradient backpropagation, DPO, GRPO and forward-process RL. The site describes a half-day tutorial on aligning image and video diffusion models with human preferences.

## Background

Post-training aligns a pretrained model to what people want. DPO learns from pairs of preferred and rejected outputs, and GRPO is an RL method that compares a group of samples to compute advantages. Applying them to diffusion models needs adaptations because generation is a multi-step denoising process.

## What I took away

- A good single source to survey diffusion alignment methods.
- Many LLM post-training ideas transfer to image and video models.

## Reference

- Original post: [https://x.com/LiangZheng_06/status/2107986518312993071](https://x.com/LiangZheng_06/status/2107986518312993071)
- Author: [@LiangZheng_06](https://x.com/LiangZheng_06) on X, posted 2026-10-08
