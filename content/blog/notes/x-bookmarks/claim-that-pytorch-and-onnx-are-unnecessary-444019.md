---
title: "Claim That PyTorch and ONNX Are Unnecessary"
publishDate: "2026-11-22"
date: "2026-11-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "tools", "gpu-kernels"]
author: "Hiep Tran"
featured: false
excerpt: "A post claims you can build inference pipelines directly in C++ with CUDA, Vulkan or Metal for speed and smaller footprint."
---

> Source: [@janusch_patas on X](https://x.com/janusch_patas/status/2124624497982444019) · posted 2026-11-22

## Summary

The author declares PyTorch and ONNX over and claims you can build an entire inference pipeline in C++ and CUDA, Vulkan or Metal with faster inference and a smaller footprint without the bloat.

## Background

PyTorch and ONNX are common frameworks for training and exporting models. Custom C++ pipelines can be lean and fast for deployment, but they cost development time and portability. The claim is a strong opinion.

## What I took away

- Custom engines can reduce footprint for fixed deployments.
- Frameworks still offer faster development.
- Treat the sweeping claim as opinion.

## Reference

- Original post: [https://x.com/janusch_patas/status/2124624497982444019](https://x.com/janusch_patas/status/2124624497982444019)
- Author: [@janusch_patas](https://x.com/janusch_patas) on X, posted 2026-11-22
