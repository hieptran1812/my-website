---
title: "vLLM Recipe For MiMo-V2-Flash"
publishDate: "2025-12-22"
date: "2025-12-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "kv-cache"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002938138549682366-1.jpg"
excerpt: "The vLLM project publishes an official serving recipe for XiaomiMiMo MiMo-V2-Flash with parallelism and KV cache tuning tips."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2002938138549682366) · posted 2025-12-22

![vLLM Recipe For MiMo-V2-Flash](/imgs/blogs/x-bookmarks/2002938138549682366-1.jpg)

## Summary

vLLM published an official recipe for serving XiaomiMiMo/MiMo-V2-Flash. It covers tool calling, DP/TP/EP configurations and the key settings that affect context length, latency and KV cache. One tip is to set --max-model-len to manage memory; commands are in the images.

## Background

DP, TP and EP stand for data, tensor and expert parallelism, ways of splitting a model over GPUs. The KV cache stores attention data for past tokens, and its size grows with context length. Capping max model length limits how much memory the cache can use.

## What I took away

- Set --max-model-len to cap KV cache memory.
- Choose DP/TP/EP configs to fit your model and hardware.
- Official recipes are a good starting point for serving new models.

## Reference

- Original post: [https://x.com/vllm_project/status/2002938138549682366](https://x.com/vllm_project/status/2002938138549682366)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2025-12-22
