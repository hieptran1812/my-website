---
title: "Quantized Qwen3.5 27B Releases For vLLM"
publishDate: "2026-03-01"
date: "2026-03-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "open-models", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2028054591430922669-1.jpg"
excerpt: "The author released INT4 and NVFP4/MXFP4 quantized versions of Qwen3.5 27B, keeping some layers in 16-bit, tuned for vLLM."
---

> Source: [@bnjmn_marie on X](https://x.com/bnjmn_marie/status/2028054591430922669) · posted 2026-03-01

![Quantized Qwen3.5 27B Releases For vLLM](/imgs/blogs/x-bookmarks/2028054591430922669-1.jpg)

## Summary

The author released quantized versions of **Qwen3.5 27B**. One set uses AutoRound (INT4) and keeps the linear attention layers in 16-bit. Another uses LLM Compressor (NVFP4/MXFP4) and also keeps the shared expert and MLP gate in 16-bit. They are optimized for fast inference with vLLM.

## Background

Quantization stores model weights with fewer bits, such as 4-bit numbers, so the model uses less memory and runs faster. Some layers are sensitive to this loss of precision, so careful recipes leave them in 16-bit. vLLM is a popular engine for serving LLMs.

## What I took away

- Quantization recipes often keep sensitive layers (attention variants, gates) in higher precision.
- INT4 and NVFP4/MXFP4 are different 4-bit formats made by different tools; compare them on your own task.
- Check that the quantized model is supported by your serving engine, here vLLM.

## Reference

- Original post: [https://x.com/bnjmn_marie/status/2028054591430922669](https://x.com/bnjmn_marie/status/2028054591430922669)
- Author: [@bnjmn_marie](https://x.com/bnjmn_marie) on X, posted 2026-03-01
