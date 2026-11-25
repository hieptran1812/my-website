---
title: "Serving TTS Differs From Serving LLMs"
publishDate: "2026-06-29"
date: "2026-06-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "audio-video"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2071427198947639757-1.jpg"
excerpt: "vLLM notes text-to-speech serving has first-audio latency, continuity and multi-stage pipeline challenges."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2071427198947639757) · posted 2026-06-29

![Serving TTS Differs From Serving LLMs](/imgs/blogs/x-bookmarks/2071427198947639757-1.jpg)

## Summary

@vllm_project says serving TTS is a different problem: a first-audio budget of a few hundred ms, continuous audio across chunks, enough concurrent streams per GPU, and a multi-stage pipeline.

## Background

TTS turns text into audio. Streaming requires steady chunks.

## What I took away

- TTS needs low first-audio latency.
- Pipelines add scheduling complexity.

## Reference

- Original post: [https://x.com/vllm_project/status/2071427198947639757](https://x.com/vllm_project/status/2071427198947639757)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-06-29
