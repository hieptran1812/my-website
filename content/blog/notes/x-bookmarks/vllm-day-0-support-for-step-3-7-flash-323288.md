---
title: "vLLM Day-0 Support for Step-3.7-Flash"
publishDate: "2026-05-29"
date: "2026-05-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "open-models", "moe"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2060155953715323288-1.jpg"
excerpt: "vLLM congratulates StepFun on Step-3.7-Flash and lists its specs and supported formats."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2060155953715323288) · posted 2026-05-29

![vLLM Day-0 Support for Step-3.7-Flash (1)](/imgs/blogs/x-bookmarks/2060155953715323288-1.jpg)

![vLLM Day-0 Support for Step-3.7-Flash (2)](/imgs/blogs/x-bookmarks/2060155953715323288-2.jpg)

## Summary

vLLM says it has day-0 support for Step-3.7-Flash, a 198B sparse MoE vision-language model with about 11B active parameters per token and a 256K context. The quoted StepFun post claims #1 on ClawEval-1.1 (67.1), #1 on SimpleVQA Search (79.2), #2 on SWE-PRO (56.3) and 95.3 on V* Python, with open weights under Apache 2.0.

## Background

FP8 and NVFP4 are low-precision formats that shrink memory and speed up inference. Benchmark ranks here are the vendor's claims.

## What I took away

- Open weights under Apache 2.0 allow commercial use.
- Low-precision formats like NVFP4 make large MoE models easier to serve.
- Verify benchmark claims independently.

## Reference

- Original post: [https://x.com/vllm_project/status/2060155953715323288](https://x.com/vllm_project/status/2060155953715323288)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-05-29
