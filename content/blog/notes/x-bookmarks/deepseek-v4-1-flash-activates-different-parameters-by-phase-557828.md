---
title: "DeepSeek V4.1 Flash Activates Different Parameters by Phase"
publishDate: "2026-09-10"
date: "2026-09-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "A reaction noting DeepSeek V4.1 Flash uses 16B active parameters on prefill and 8B on decode, like an encoder-decoder."
---

> Source: [@nrehiew_ on X](https://x.com/nrehiew_/status/2097934528006557828) · posted 2026-09-10

## Summary

The author says welcome back encoder-decoder, noting the model is 552B total with 8B active on decode and 16B active on prefill. They say they have never seen one model activate a different number of parameters in different phases.

## Background

An encoder-decoder model uses one network to read the input and another to generate output, unlike decoder-only LLMs. Prefill processes the prompt in parallel and is compute bound; decode generates one token at a time and is memory bound. Using more parameters for prefill and fewer for decode matches these different costs.

## What I took away

- Prefill and decode have different bottlenecks, and architectures can exploit that.
- Older ideas like encoder-decoder can return in new forms.

## Reference

- Original post: [https://x.com/nrehiew_/status/2097934528006557828](https://x.com/nrehiew_/status/2097934528006557828)
- Author: [@nrehiew_](https://x.com/nrehiew_) on X, posted 2026-09-10
