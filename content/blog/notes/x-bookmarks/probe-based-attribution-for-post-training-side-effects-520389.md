---
title: "Probe-Based Attribution For Post-Training Side Effects"
publishDate: "2026-04-29"
date: "2026-04-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "interpretability", "ai-safety"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2049547521941520389-1.jpg"
excerpt: "Goodfire shares research tracing undesired post-training side effects to data with probes, claiming a 63% reduction."
---

> Source: [@GoodfireAI on X](https://x.com/GoodfireAI/status/2049547521941520389) · posted 2026-04-29

![Probe-Based Attribution For Post-Training Side Effects](/imgs/blogs/x-bookmarks/2049547521941520389-1.jpg)

## Summary

Goodfire announces research on finding and reducing undesired side effects of LLM post-training with probe-based data attribution. The quoted thread says probes can trace a harmful effect to specific training data, and filtering it cuts the behavior by 63%, beating LLM-judge and gradient attribution at 10x lower cost.

## Background

Data attribution finds which training examples cause a behavior. A probe is a small classifier on internal activations. Cheaper attribution makes cleaning training data practical. Numbers are the authors' claims.

## What I took away

- Probes can link behaviors to specific training data.
- Filtering causal data can beat broad methods on cost.

## Reference

- Original post: [https://x.com/GoodfireAI/status/2049547521941520389](https://x.com/GoodfireAI/status/2049547521941520389)
- Author: [@GoodfireAI](https://x.com/GoodfireAI) on X, posted 2026-04-29
