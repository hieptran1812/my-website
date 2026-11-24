---
title: "vLLM Day Zero Deployment Guide for Kimi K3"
publishDate: "2026-07-27"
date: "2026-07-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "open-models", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM publishes a deployment guide for Kimi K3, a 2.8T-parameter MoE with 1M context and native vision, supported from day 0."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2081788542079123939) · posted 2026-07-27

## Summary

vLLM says **Kimi K3** is here: a 2.8T-parameter MoE with a 1M-token context and native vision. It runs on vLLM from day 0. vLLM shares a canonical deployment guide covering architecture, kernels, recipes and production flags.

## Background

Day 0 support means the model works in the serving engine on release day. Large MoE models need special settings for parallelism, memory and kernels. A canonical guide saves teams from guessing these settings.

## What I took away

- Start with the official engine guide when deploying a new large model.
- Production flags matter for memory and speed; do not rely on defaults blindly.
- 1M context needs careful KV and memory planning.

## Reference

- Original post: [https://x.com/vllm_project/status/2081788542079123939](https://x.com/vllm_project/status/2081788542079123939)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-07-27
