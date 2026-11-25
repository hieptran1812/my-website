---
title: "Supervising Early Tokens In On-Policy Distillation"
publishDate: "2026-06-27"
date: "2026-06-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2070870870088970525-1.jpg"
excerpt: "A theoretical revisit of on-policy distillation finds that supervising only the first 30% of tokens nearly matches full OPD."
---

> Source: [@sheriyuo on X](https://x.com/sheriyuo/status/2070870870088970525) · posted 2026-06-27

![Supervising Early Tokens In On-Policy Distillation](/imgs/blogs/x-bookmarks/2070870870088970525-1.jpg)

## Summary

@sheriyuo calls this a nice theoretical revisit of OPD. With the same supervision budget, supervising only the first 30% of tokens nearly matches standard OPD, while supervising only the last 30% barely works. The paper frames OPD as a constrained optimization problem. The text is cut off after that.

## Background

On-policy distillation has a teacher guide the student on the student's own samples, token by token. Supervising only part of each sequence saves compute. Early tokens seem to matter more, likely because they shape the rest of the sequence.

## What I took away

- Early tokens carry most of the useful supervision signal in this result.
- This could cut distillation cost if it holds in your setup.

## Reference

- Original post: [https://x.com/sheriyuo/status/2070870870088970525](https://x.com/sheriyuo/status/2070870870088970525)
- Author: [@sheriyuo](https://x.com/sheriyuo) on X, posted 2026-06-27
