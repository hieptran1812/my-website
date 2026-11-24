---
title: "Spark X2.5 On-Device Models With 1M Context"
publishDate: "2026-09-01"
date: "2026-09-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "llm-architecture", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "SparkLLM open-sources Spark X2.5-4B and 1.7B, compact on-device agent models with native 1M-token context and day-0 vLLM support."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2094674692133298363) · posted 2026-09-01

## Summary

vLLM congratulates SparkLLM on **Spark-X2.5-4B** and **1.7B**, compact on-device agent models covering 200+ languages with native 1M context, supported in vLLM from day one through an out-of-tree plugin. The quoted SparkLLM post claims they are the only on-device models with native context up to 1,000,000 tokens and uses a hybrid-attention architecture tuned for agents, code, math and instruction following.

## Background

On-device models are small enough to run on phones or laptops. Very long context is hard for small models because standard attention memory grows with length. Hybrid attention mixes cheaper attention types with full attention to make long context affordable. An out-of-tree plugin lets a model run in vLLM without changing vLLM's main code.

## What I took away

- Hybrid attention is the common trick for giving small models very long context.
- Out-of-tree plugins let new architectures reach serving engines quickly.
- Test long-context quality yourself, since supporting 1M tokens is not the same as using them well.

## Reference

- Original post: [https://x.com/vllm_project/status/2094674692133298363](https://x.com/vllm_project/status/2094674692133298363)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-09-01
