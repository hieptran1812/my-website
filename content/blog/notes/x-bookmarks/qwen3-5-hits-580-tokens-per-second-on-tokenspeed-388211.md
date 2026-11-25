---
title: "Qwen3.5 Hits 580 Tokens Per Second on TokenSpeed"
publishDate: "2026-05-27"
date: "2026-05-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving", "gpu-kernels"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2059666448998388211-1.jpg"
excerpt: "PyTorch highlights a record 580 tokens per second for Qwen3.5 agentic workloads on the TokenSpeed engine."
---

> Source: [@PyTorch on X](https://x.com/PyTorch/status/2059666448998388211) · posted 2026-05-27

![Qwen3.5 Hits 580 Tokens Per Second on TokenSpeed](/imgs/blogs/x-bookmarks/2059666448998388211-1.jpg)

## Summary

PyTorch says speed-of-light optimization for Qwen3.5 on the TokenSpeed inference engine reached a **record-breaking 580 tokens per second** for agentic workloads on NVIDIA GPUs. It points to a PyTorch Foundation community blog post for details.

## Background

Agentic workloads generate many sequential tokens, so per-user speed matters more than total throughput. Speed-of-light means pushing close to hardware limits. The 580 tps figure is reported by the project.

## What I took away

- Single-stream speed is key for agents.
- Read the blog for the optimizations used.
- Check the batch size and hardware behind the record.

## Reference

- Original post: [https://x.com/PyTorch/status/2059666448998388211](https://x.com/PyTorch/status/2059666448998388211)
- Author: [@PyTorch](https://x.com/PyTorch) on X, posted 2026-05-27
