---
title: "HF Blog On gpt-oss Speed Tricks"
publishDate: "2025-12-24"
date: "2025-12-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "quantization", "moe"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2003890268597039200-1.jpg"
excerpt: "A post highlights a Hugging Face blog on the techniques that make gpt-oss run fast in transformers."
---

> Source: [@Hesamation on X](https://x.com/Hesamation/status/2003890268597039200) · posted 2025-12-24

![HF Blog On gpt-oss Speed Tricks](/imgs/blogs/x-bookmarks/2003890268597039200-1.jpg)

## Summary

Hesamation recommends a blog by Hugging Face developers that lists the tricks from OpenAI's gpt-oss that speed up transformers. It covers MXFP4 quantization, tensor parallelism, expert parallelism and dynamic sliding window, among others. The quoted post is a list of blogs by Sergio Paniego, including faster transformers.

## Background

MXFP4 is a 4-bit floating point format that shrinks weights. Tensor parallelism splits a layer across GPUs, and expert parallelism spreads the experts of an MoE model across devices. Sliding window attention limits how far back tokens look.

## What I took away

- Fast inference combines quantization, parallelism and attention tweaks.
- Read the blog to see these used in real code.

## Reference

- Original post: [https://x.com/Hesamation/status/2003890268597039200](https://x.com/Hesamation/status/2003890268597039200)
- Author: [@Hesamation](https://x.com/Hesamation) on X, posted 2025-12-24
