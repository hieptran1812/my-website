---
title: "Treating Long RL Runs as a Live Service"
publishDate: "2026-11-23"
date: "2026-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2125008811039797507-1.jpg"
excerpt: "A researcher shares a blog on Microsoft AI's Rocket RL infrastructure and a shift toward treating long RL runs as a live service."
---

> Source: [@kylelostat on X](https://x.com/kylelostat/status/2125008811039797507) · posted 2026-11-23

![Treating Long RL Runs as a Live Service](/imgs/blogs/x-bookmarks/2125008811039797507-1.jpg)

## Summary

The author says an interesting shift from early research days was treating long RL runs as a live service. In that world, data research looks different because hard data goes stale quickly as the model improves. The quoted post says reaching the frontier needs RL runs lasting weeks to months and introduces the in-house infrastructure called Rocket.

## Background

Reinforcement learning for LLMs trains a model with reward signals from tasks. When runs last weeks, the system must keep running, recover from failures and keep feeding fresh tasks. Tasks that were hard at the start become easy as the model improves, so they stop teaching anything.

## What I took away

- Long RL runs need operations thinking like a production service.
- Training data can go stale as the model gets better.
- Keep a pipeline that refreshes hard tasks.

## Reference

- Original post: [https://x.com/kylelostat/status/2125008811039797507](https://x.com/kylelostat/status/2125008811039797507)
- Author: [@kylelostat](https://x.com/kylelostat) on X, posted 2026-11-23
