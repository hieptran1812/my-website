---
title: "Marin 535B-A23B Training Progress and Open Panel"
publishDate: "2026-08-28"
date: "2026-08-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "moe", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Percy Liang reports Marin 535B-A23B is about 7% through training and announces a Zoom panel on its design decisions."
---

> Source: [@percyliang on X](https://x.com/percyliang/status/2093139094968922440) · posted 2026-08-28

## Summary

Percy Liang says **Marin 535B-A23B** is about 7% done training and looks on track. He announces a Zoom panel on Tuesday, Sept 1 at 10 PT where the Marin team will discuss the design decisions, tradeoffs and lessons from the run.

## Background

The name 535B-A23B means a mixture-of-experts model with 535 billion total parameters but about 23 billion active per token. MoE models route each token to a few experts, giving large capacity at lower compute. Open discussion of a training run in progress is rare.

## What I took away

- Open labs sharing live training decisions is a good learning opportunity.
- MoE naming like 535B-A23B separates total from active parameters.
- Look for a recording of the panel if you missed it.

## Reference

- Original post: [https://x.com/percyliang/status/2093139094968922440](https://x.com/percyliang/status/2093139094968922440)
- Author: [@percyliang](https://x.com/percyliang) on X, posted 2026-08-28
