---
title: "Diffusion Cache Acceleration In vLLM-Omni"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002233999947890810-1.jpg"
excerpt: "vLLM-Omni adds TeaCache and Cache-DiT backends to reuse computation between diffusion steps without retraining."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2002233999947890810) · posted 2025-12-20

![Diffusion Cache Acceleration In vLLM-Omni](/imgs/blogs/x-bookmarks/2002233999947890810-1.jpg)

## Summary

The vLLM project says diffusion serving is costly because each image needs dozens of timesteps with a lot of repeated compute between adjacent steps. vLLM-Omni now supports TeaCache and Cache-DiT backends that reuse intermediate Transformer computations. The post says this needs no retraining.

## Background

A diffusion model makes an image by repeatedly denoising it over many steps. Neighbouring steps often produce similar internal values, so caching and reusing some of them saves work. This lowers cost and latency at inference time, but can slightly change quality.

## What I took away

- Diffusion cost scales with the number of timesteps, so reuse between steps is a big lever.
- Cache-based speedups need no retraining, which makes them easy to try.
- Check image quality after enabling caching, not only speed.

## Reference

- Original post: [https://x.com/vllm_project/status/2002233999947890810](https://x.com/vllm_project/status/2002233999947890810)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2025-12-20
