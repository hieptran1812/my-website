---
title: "vLLM AFD Plugin for Attention-FFN Disaggregation"
publishDate: "2026-07-24"
date: "2026-07-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "moe", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM announces an experimental plugin that brings Attention-FFN Disaggregation to MoE serving, built by several teams."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2080673192939807083) · posted 2026-07-24

## Summary

vLLM congratulates the teams behind the **vLLM AFD Plugin**, including Ascend and vLLM, StepFun, Ant Group and FastAFD. It is a new experimental plugin under vllm-project that brings Attention-FFN Disaggregation to MoE serving. The post notes that attention and the expert/FFN path are two very different workloads (text cut off).

## Background

In a transformer, attention is memory-bound and depends on the KV cache, while the FFN or expert layers are compute-heavy matrix math. Disaggregation runs them on separate groups of GPUs so each can be sized and tuned for its own workload. This is similar to splitting prefill and decode.

## What I took away

- Splitting different workloads onto separate hardware can improve efficiency.
- AFD is experimental; test carefully before production use.
- Watch for AFD support in other engines too.

## Reference

- Original post: [https://x.com/vllm_project/status/2080673192939807083](https://x.com/vllm_project/status/2080673192939807083)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-07-24
