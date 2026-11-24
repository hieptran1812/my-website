---
title: "FastH3 Brings Fast Open Video Generation to vLLM-Omni"
publishDate: "2026-08-28"
date: "2026-08-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "audio-video", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM congratulates the FastVideo team on FastH3, an accelerated MiniMax H3 that generates 15s of 768p video in 13s."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2093430388391825797) · posted 2026-08-28

## Summary

vLLM congratulates the FastVideo team at Hao AI Lab on the **FastH3** release, says it sponsored the effort, and says it is working to bring FastH3 into vLLM-Omni. The quoted thread says FastH3 v1, built on open-weight MiniMax H3, generates 15 seconds of 768p video in 13 seconds with up to 14x speedup on NVIDIA Blackwell GPUs and is fully open so others can improve the recipe.

## Background

Video diffusion models usually need many denoising steps, each expensive. Acceleration recipes cut steps through distillation and use sparse attention to reduce cost per step. vLLM-Omni is the vLLM effort to serve non-text models like video.

## What I took away

- Open acceleration recipes let the community keep improving video speed.
- Near real-time video is becoming possible on current GPUs.
- Compare quality samples, since speedups often trade some fidelity.

## Reference

- Original post: [https://x.com/vllm_project/status/2093430388391825797](https://x.com/vllm_project/status/2093430388391825797)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-08-28
