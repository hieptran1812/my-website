---
title: "vLLM v0.15.0 Release Highlights"
publishDate: "2026-01-31"
date: "2026-01-31"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2017421213392638321-1.jpg"
excerpt: "vLLM v0.15.0 ships with async scheduling, pipeline parallelism, Mamba prefix caching and Blackwell FP4 gains."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2017421213392638321) · posted 2026-01-31

![vLLM v0.15.0 Release Highlights](/imgs/blogs/x-bookmarks/2017421213392638321-1.jpg)

## Summary

vLLM announces v0.15.0 with 335 commits from 158 contributors. Highlights include async scheduling with pipeline parallelism, Mamba prefix caching (~2x speedup), 65% faster Blackwell FP4, AMD RDNA3/RDNA4 support, and models like Kimi-K2.5 and EAGLE3.

## Background

Prefix caching reuses computation for shared prompt starts. FP4 is a 4-bit number format on NVIDIA Blackwell GPUs. Pipeline parallelism splits layers across GPUs.

## What I took away

- Check release notes before upgrading.
- New hardware support widens options.
- Speedups are reported by the project.

## Reference

- Original post: [https://x.com/vllm_project/status/2017421213392638321](https://x.com/vllm_project/status/2017421213392638321)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-01-31
