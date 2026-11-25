---
title: "Caveat On Energy-Based Models Scaling Claim"
publishDate: "2026-11-06"
date: "2026-11-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2118822462888587728-1.jpg"
excerpt: "A critique notes the TEM paper uses a stronger backbone, making the comparison with diffusion unclear."
---

> Source: [@WangYw251 on X](https://x.com/WangYw251/status/2118822462888587728) · posted 2026-11-06

![Caveat On Energy-Based Models Scaling Claim (1)](/imgs/blogs/x-bookmarks/2118822462888587728-1.jpg)

![Caveat On Energy-Based Models Scaling Claim (2)](/imgs/blogs/x-bookmarks/2118822462888587728-2.jpg)

## Summary

The author says TEM uses LightningDiT (DiT with RoPE, RMS norm and SwiGLU) as the backbone but compares against vanilla DiT and SiT, so it is hard to tell whether EBMs scale better or the architecture is stronger. The quoted post claims TEM outscales diffusion without guidance.

## Background

A fair comparison keeps the backbone the same. Otherwise gains may come from the architecture, not the method.

## What I took away

- Check baselines use the same backbone.
- Confounders hide true gains.
- Scaling claims need controlled tests.

## Reference

- Original post: [https://x.com/WangYw251/status/2118822462888587728](https://x.com/WangYw251/status/2118822462888587728)
- Author: [@WangYw251](https://x.com/WangYw251) on X, posted 2026-11-06
