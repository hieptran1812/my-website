---
title: "vLLM Day-0 Support for DeepSeek-V4.1-Flash"
publishDate: "2026-09-10"
date: "2026-09-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "moe", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM serves DeepSeek-V4.1-Flash from day 0 on NVIDIA and AMD, a 552B MoE with vision and 1M context."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2097940813242405272) · posted 2026-09-10

## Summary

DeepSeek introduces **DeepSeek-V4.1-Flash**, the smallest model in its new architecture family, with native visual understanding and a focus on speed and throughput. vLLM says it supports it from day 0, verified on NVIDIA and AMD GPUs. They describe a 552B MoE backbone with 1M context, 8B active parameters while reading the prompt and 16B while writing.

## Background

MoE models activate only some experts per token, so active parameters are much smaller than total. Day-0 support means the serving engine works the day the model is released. Different active sizes for reading (prefill) and writing (decode) is unusual.

## What I took away

- Check active parameters, not total, to estimate compute cost.
- Day-0 support in vLLM makes new open models usable right away.
- Separate prefill and decode costs when planning serving capacity.

## Reference

- Original post: [https://x.com/vllm_project/status/2097940813242405272](https://x.com/vllm_project/status/2097940813242405272)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-09-10
