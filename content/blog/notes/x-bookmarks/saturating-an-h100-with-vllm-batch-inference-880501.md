---
title: "Saturating An H100 With vLLM Batch Inference"
publishDate: "2026-01-14"
date: "2026-01-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2011585247297880501-1.jpg"
excerpt: "vLLM highlights a Modal guide for maxing H100 throughput with Qwen 3 8B, claiming 5x cheaper than APIs."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2011585247297880501) · posted 2026-01-14

![Saturating An H100 With vLLM Batch Inference](/imgs/blogs/x-bookmarks/2011585247297880501-1.jpg)

## Summary

vLLM shares a guide showing how to reach full H100 utilization with Qwen 3 8B using the FlashInfer backend, async scheduling and tuned batch sizes. The quoted post claims about 30k input and 2k output tokens per second per H100, 5x cheaper than APIs for summarizing SEC filings.

## Background

Batch workloads do not need low latency, so the server can run big batches to keep the GPU busy and cut cost per token.

## What I took away

- Batch jobs favor throughput over latency.
- Self-hosting can beat API prices at high utilization.

## Reference

- Original post: [https://x.com/vllm_project/status/2011585247297880501](https://x.com/vllm_project/status/2011585247297880501)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-01-14
