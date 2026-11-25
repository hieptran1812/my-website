---
title: "vLLM Day-0 Support for Qwen3-Coder-Next"
publishDate: "2026-02-03"
date: "2026-02-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2018742511502856568-1.jpg"
excerpt: "vLLM 0.15.0 supports Qwen3-Coder-Next from day one, an 80B MoE with 3B active parameters for coding agents."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2018742511502856568) · posted 2026-02-03

![vLLM Day-0 Support for Qwen3-Coder-Next (1)](/imgs/blogs/x-bookmarks/2018742511502856568-1.jpg)

![vLLM Day-0 Support for Qwen3-Coder-Next (2)](/imgs/blogs/x-bookmarks/2018742511502856568-2.jpg)

## Summary

vLLM congratulates Alibaba Qwen on Qwen3-Coder-Next and says day-0 support is in vLLM 0.15.0. It describes an 80B MoE with 3B active parameters, claimed to match models 10-20x larger. The quoted Qwen post cites 800K verifiable tasks with executable environments and strong SWE-Bench Pro results.

## Background

In a mixture-of-experts model only a few experts run per token, so active parameters are far fewer than total. Day-0 support means the inference engine can serve the model on release. Verifiable tasks are ones where code can be run to check the answer.

## What I took away

- Active parameters decide inference cost in MoE models.
- Executable environments are a strong training signal for coding agents.
- Treat 'matches 10-20x larger' as a claim to test.

## Reference

- Original post: [https://x.com/vllm_project/status/2018742511502856568](https://x.com/vllm_project/status/2018742511502856568)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-02-03
