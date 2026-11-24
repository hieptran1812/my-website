---
title: "SGLang Weight Cache Daemon Cuts 1T Model Restart Time"
publishDate: "2026-08-22"
date: "2026-08-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "SGLang introduced a Weight Cache Daemon that reduces restart time for a 1T-parameter model from minutes to seconds."
---

> Source: [@sgl_project on X](https://x.com/sgl_project/status/2091048937558085862) · posted 2026-08-22

## Summary

SGLang says restarting a 1T model used to take 8.8 minutes and now takes 32 seconds, thanks to a collaboration with AntLingAGI and Alibaba. The quoted announcement explains that on Ling-2.6-1T FP8, the Weight Cache Daemon reduced weight loading to about 0.63 seconds, up to about 780 times faster than loading from disk, and cut total engine startup to about 0.53 minutes. A full write-up is on the LMSYS blog.

## Background

Before a model can serve requests, its weights (often hundreds of gigabytes for very large models) must be loaded into GPU memory. Reading them from disk is slow, so keeping them cached in a separate long-lived process lets a crashed or updated engine come back almost instantly. Fast recovery matters for reliability and for rolling updates in production.

## What I took away

- Startup and recovery time is a real cost in serving very large models, not just tokens per second.
- Separating weight storage from the engine process is a pattern worth considering for fast restarts.
- Read the write-up to see what conditions the speedup numbers depend on.

## Reference

- Original post: [https://x.com/sgl_project/status/2091048937558085862](https://x.com/sgl_project/status/2091048937558085862)
- Author: [@sgl_project](https://x.com/sgl_project) on X, posted 2026-08-22
