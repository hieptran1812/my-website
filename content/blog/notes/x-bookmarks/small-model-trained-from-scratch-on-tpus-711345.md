---
title: "Small Model Trained From Scratch on TPUs"
publishDate: "2025-12-16"
date: "2025-12-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "pretraining", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001079075075711345-1.jpg"
excerpt: "A solo developer shows a model trained from scratch on TPUs with the Neuroblast-v3 architecture, running on local vLLM."
---

> Source: [@mkurman88 on X](https://x.com/mkurman88/status/2001079075075711345) · posted 2025-12-16

![Small Model Trained From Scratch on TPUs (1)](/imgs/blogs/x-bookmarks/2001079075075711345-1.jpg)

![Small Model Trained From Scratch on TPUs (2)](/imgs/blogs/x-bookmarks/2001079075075711345-2.png)

## Summary

The author describes a single-person project trained from scratch on TPUs from the Google TRC program. It uses the SYTNH dataset from @pleiasfr and a Neuroblast-v3 architecture, and it runs on their local vLLM instance. They say they are amazed at how good it looks and how fast it is.

## Background

vLLM is a popular open-source engine for serving language models. TRC gives researchers free access to Google TPUs. Training a model from scratch, rather than fine-tuning one, is rare for individuals, so this shows how reachable it has become.

## What I took away

- Free compute programs like TPU Research Cloud can enable solo pretraining projects.
- A custom architecture can be served with vLLM for quick testing.
- The post gives no benchmarks, so quality claims are unverified.

## Reference

- Original post: [https://x.com/mkurman88/status/2001079075075711345](https://x.com/mkurman88/status/2001079075075711345)
- Author: [@mkurman88](https://x.com/mkurman88) on X, posted 2025-12-16
