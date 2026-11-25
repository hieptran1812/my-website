---
title: "Article On How LLM Inference Works"
publishDate: "2025-11-22"
date: "2025-11-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "kv-cache", "quantization", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1992211414623322116-1.png"
excerpt: "Arpit Bhayani shares an article covering the full LLM inference path from embeddings to KV caching and quantization."
---

> Source: [@arpit_bhayani on X](https://x.com/arpit_bhayani/status/1992211414623322116) · posted 2025-11-22

![Article On How LLM Inference Works](/imgs/blogs/x-bookmarks/1992211414623322116-1.png)

## Summary

The author wrote an article after about a week of building understanding of how LLM inference works. It covers embeddings, attention, KV caching, quantization and more. They say they are no expert but hope it helps others.

## Background

Inference is the process of running a trained model to generate text. The KV cache stores past attention keys and values so they are not recomputed for each new token. Quantization uses lower-precision numbers to cut memory and speed up serving.

## What I took away

- A good mental model of inference starts with attention and the KV cache.
- Quantization trades a little accuracy for memory and speed.
- Writing a distilled summary is a good way to learn a topic.

## Reference

- Original post: [https://x.com/arpit_bhayani/status/1992211414623322116](https://x.com/arpit_bhayani/status/1992211414623322116)
- Author: [@arpit_bhayani](https://x.com/arpit_bhayani) on X, posted 2025-11-22
