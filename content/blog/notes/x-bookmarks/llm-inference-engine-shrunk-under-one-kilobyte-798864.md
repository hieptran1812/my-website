---
title: "LLM Inference Engine Shrunk Under One Kilobyte"
publishDate: "2026-11-04"
date: "2026-11-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "A developer shrinks a dependency-free inference engine running SmolLM2-135M from 2.9 KB to 1023 bytes."
---

> Source: [@cneuralnetwork on X](https://x.com/cneuralnetwork/status/2117871577102798864) · posted 2026-11-04

## Summary

The quoted post shows an inference engine of about 2.9 KB that runs a real model, SmolLM2-135M, with no Python, no libraries and not even libc, small enough to fit in a QR code. @cneuralnetwork says he worked overnight and got it under 1 KB (1023 bytes), with a pull request linked.

## Background

An inference engine is the code that loads model weights and computes outputs. Most engines are large because of libraries and optimizations. A tiny one shows the core math of running a transformer is simple; the weights themselves are stored separately and are much larger.

## What I took away

- The core logic of running a transformer is very small.
- Code golf projects are a fun way to understand what an engine really needs.
- Size of the engine is not the size of the model; weights are separate.

## Reference

- Original post: [https://x.com/cneuralnetwork/status/2117871577102798864](https://x.com/cneuralnetwork/status/2117871577102798864)
- Author: [@cneuralnetwork](https://x.com/cneuralnetwork) on X, posted 2026-11-04
