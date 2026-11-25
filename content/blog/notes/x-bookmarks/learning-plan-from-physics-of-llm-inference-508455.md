---
title: "Learning Plan From Physics Of LLM Inference"
publishDate: "2026-02-08"
date: "2026-02-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "gpu-kernels", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2020376708017508455-1.jpg"
excerpt: "The author starts a book on LLM inference covering GEMM, CUDA, FlashAttention, batching and serving."
---

> Source: [@JaydevTonde on X](https://x.com/JaydevTonde/status/2020376708017508455) · posted 2026-02-08

![Learning Plan From Physics Of LLM Inference (1)](/imgs/blogs/x-bookmarks/2020376708017508455-1.jpg)

![Learning Plan From Physics Of LLM Inference (2)](/imgs/blogs/x-bookmarks/2020376708017508455-2.jpg)

## Summary

The author is starting the Physics of LLM Inference book by Elliot Arledge. The plan covers GEMM, GEMV, Hello CUDA, kernel optimization, Flash Attention, continuous batching, advanced scheduling, MoE and a production server.

## Background

GEMM and GEMV are matrix multiplications that dominate LLM compute. FlashAttention is a memory-efficient attention kernel. Continuous batching mixes requests to keep GPUs busy. Together these form the core of fast inference.

## What I took away

- Inference speed starts with matrix math and memory.
- The chapter list is a good study roadmap.
- Build up from kernels to serving.

## Reference

- Original post: [https://x.com/JaydevTonde/status/2020376708017508455](https://x.com/JaydevTonde/status/2020376708017508455)
- Author: [@JaydevTonde](https://x.com/JaydevTonde) on X, posted 2026-02-08
