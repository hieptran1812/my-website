---
title: "vLLM Reaches 464 Tokens per Second on Kimi-K3"
publishDate: "2026-07-29"
date: "2026-07-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "speculative-decoding", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM reports a peak batch-size-1 decode of 464 tok/s on Kimi-K3 with DSpark on GB300 GPUs under a low-entropy workload."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2082267336279814173) · posted 2026-07-29

## Summary

vLLM says it hit a new peak **batch size 1 decode of 464 tok/s** on Kimi-K3. This was under a low-entropy reasoning workload, using DSpark on vLLM with 4x4 GB300. It says the benchmark is reproducible with the public image vllm/vllm-openai:kimi-k3 and Inferact's DSpark.

## Background

Batch size 1 decode measures speed for a single request, which reflects latency. Low-entropy workloads have predictable text, which makes speculative decoding drafts more likely to be accepted, raising speed. DSpark appears to be the speculative drafter used here; GB300 is an NVIDIA Blackwell Ultra GPU.

## What I took away

- Speculative decoding speed depends heavily on how predictable the text is.
- Compare headline numbers only under the same workload type.
- Reproducible public images make benchmarks easier to verify.

## Reference

- Original post: [https://x.com/vllm_project/status/2082267336279814173](https://x.com/vllm_project/status/2082267336279814173)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-07-29
