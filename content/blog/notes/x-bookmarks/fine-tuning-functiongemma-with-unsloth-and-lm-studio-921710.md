---
title: "Fine Tuning FunctionGemma With Unsloth And LM Studio"
publishDate: "2025-12-23"
date: "2025-12-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "tools", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2003490499101921710-1.jpg"
excerpt: "LM Studio and Unsloth publish a guide to fine tune FunctionGemma and run it locally."
---

> Source: [@lmstudio on X](https://x.com/lmstudio/status/2003490499101921710) · posted 2025-12-23

![Fine Tuning FunctionGemma With Unsloth And LM Studio](/imgs/blogs/x-bookmarks/2003490499101921710-1.jpg)

## Summary

LM Studio says it worked with Unsloth on a beginner guide to fine tune FunctionGemma and run it locally. The steps are training it for custom tool calls, converting it to GGUF and importing it into LM Studio, and serving it locally for use in code. A step-by-step notebook is provided.

## Background

A tool call is when a model outputs a structured request to run a function. GGUF is a file format for running models locally with tools like llama.cpp and LM Studio. Fine tuning on your own tool schema can make a small model reliable for a narrow job.

## What I took away

- You can fine tune a small model for your own tool calls and run it locally.
- Converting to GGUF is the step that makes it usable in LM Studio.

## Reference

- Original post: [https://x.com/lmstudio/status/2003490499101921710](https://x.com/lmstudio/status/2003490499101921710)
- Author: [@lmstudio](https://x.com/lmstudio) on X, posted 2025-12-23
