---
title: "d3-turbo Multimodal Decision Model for Routing"
publishDate: "2026-10-20"
date: "2026-10-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM highlights d3-turbo, a multimodal decision model for intelligent routing that claims a top benchmark score and a large speedup."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2112352859937439745) · posted 2026-10-20

## Summary

@vllm_project says the vLLM ecosystem keeps growing and highlights d3-turbo, a multimodal decision model for intelligent routing from the vllm-sr team, claimed #1 on the Jev Decision Index and 15x faster than d3 flash with StateOnce. The quoted post adds that it can point with boxes, follow and see images in questions, and is on Hugging Face.

## Background

A routing or decision model looks at a request and decides where to send it, for example to a cheaper or stronger model. Making the router fast matters because it runs on every request.

## What I took away

- Routers can cut cost by sending easy requests to cheaper models.
- Router speed adds directly to request latency.
- Benchmark rankings are the team's claims.

## Reference

- Original post: [https://x.com/vllm_project/status/2112352859937439745](https://x.com/vllm_project/status/2112352859937439745)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-10-20
