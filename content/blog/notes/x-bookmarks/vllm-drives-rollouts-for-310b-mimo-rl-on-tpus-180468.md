---
title: "vLLM Drives Rollouts for 310B MiMo RL on TPUs"
publishDate: "2026-09-22"
date: "2026-09-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "serving", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM highlights its role in full-parameter RL of MiMo-V2.6 310B across 1,000+ TPUs, with rollouts bitwise matched to the trainer."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2102264807630180468) · posted 2026-09-22

## Summary

The vLLM project highlights Peano running full-parameter RL for Xiaomi's 310B **MiMo-V2.6** on 1,000+ TPUs. vLLM generates the rollouts, bitwise matched with the trainer in validation, and all 310B parameters move across the ICI fabric in under 2 seconds. The quoted post adds that scaling with JAX was a config change.

## Background

In RL for LLMs, the trainer updates weights and the inference engine generates new samples with them, so weights must be copied often. ICI is the inter-chip interconnect linking TPUs. Fast weight sync keeps the rollout engine from idling.

## What I took away

- Weight transfer speed between trainer and inference engine is a key RL bottleneck.
- Bitwise matching is a strong correctness check for RL pipelines.
- General serving engines like vLLM are now part of training stacks.

## Reference

- Original post: [https://x.com/vllm_project/status/2102264807630180468](https://x.com/vllm_project/status/2102264807630180468)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-09-22
