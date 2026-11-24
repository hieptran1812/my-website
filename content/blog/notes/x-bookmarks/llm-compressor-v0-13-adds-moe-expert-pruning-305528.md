---
title: "LLM Compressor v0.13 Adds MoE Expert Pruning"
publishDate: "2026-08-12"
date: "2026-08-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "moe", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "LLM Compressor v0.13.0 adds REAP expert pruning for MoE models and arbitrary bit-width quantization."
---

> Source: [@RedHat_AI on X](https://x.com/RedHat_AI/status/2087519343349305528) · posted 2026-08-12

## Summary

Red Hat AI announced LLM Compressor v0.13.0. It adds REAP expert pruning, which shrinks MoE models by removing whole experts based on calibration saliency, then quantizing what remains to FP8 or NVFP4. It also adds arbitrary bit-width quantization (3, 5, 6 and 7-bit with no wasted bits) and new weight-activation schemes.

## Background

Pruning removes parts of a model to make it smaller; with MoE you can drop entire experts that rarely matter. Calibration saliency means measuring importance on a sample dataset. Quantization stores numbers in fewer bits. Combining both can shrink a model a lot, at some risk to quality.

## What I took away

- For MoE models, removing whole experts is a structured way to cut memory.
- Calibration data should match your real workload, or pruning may remove useful experts.
- Evaluate quality after combined pruning and quantization.

## Reference

- Original post: [https://x.com/RedHat_AI/status/2087519343349305528](https://x.com/RedHat_AI/status/2087519343349305528)
- Author: [@RedHat_AI](https://x.com/RedHat_AI) on X, posted 2026-08-12
