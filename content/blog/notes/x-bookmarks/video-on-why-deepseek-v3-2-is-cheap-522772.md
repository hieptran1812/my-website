---
title: "Video On Why DeepSeek v3.2 Is Cheap"
publishDate: "2025-12-01"
date: "2025-12-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-architecture", "llm-inference", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1995559075900522772-1.jpg"
excerpt: "A video explainer on attention, KV caching and efficiency tricks like MQA, GQA, MLA and DSA behind DeepSeek v3.2."
---

> Source: [@jbhuang0604 on X](https://x.com/jbhuang0604/status/1995559075900522772) · posted 2025-12-01

![Video On Why DeepSeek v3.2 Is Cheap](/imgs/blogs/x-bookmarks/1995559075900522772-1.jpg)

## Summary

The author asks how DeepSeek v3.2 rivals top models such as GPT5 and Gemini 3 pro while being about 30x cheaper, which is their claim. Their video covers attention, why KV caching is needed, and ideas to improve attention: MQA, GQA, MLA and DSA.

## Background

The KV cache stores past keys and values so the model does not recompute them for each new token, but it uses lots of memory. MQA, GQA and MLA share or compress keys and values to shrink it. DSA is a sparse attention approach named in the post. Smaller caches mean cheaper serving.

## What I took away

- Attention cost and KV cache size drive inference price.
- Learn the progression MQA to GQA to MLA.
- The 30x figure is the author's claim.

## Reference

- Original post: [https://x.com/jbhuang0604/status/1995559075900522772](https://x.com/jbhuang0604/status/1995559075900522772)
- Author: [@jbhuang0604](https://x.com/jbhuang0604) on X, posted 2025-12-01
