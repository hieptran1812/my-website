---
title: "Choosing a Speculative Decoding Method in vLLM"
publishDate: "2026-08-28"
date: "2026-08-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A vLLM blog compares MTP, EAGLE-3, DFlash, DSpark and another method, and explains how to pick and tune them."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2093148358143795254) · posted 2026-08-28

## Summary

vLLM publishes a blog asking which speculative decoding method to use: MTP, EAGLE-3, DFlash or DSpark. It says there is no universal winner, since the best choice changes with the model, workload and speculation depth. The blog explains how five methods work and how to enable and tune them in vLLM.

## Background

Speculative decoding means a cheap drafter proposes several tokens and the main model checks them in one pass. MTP uses extra prediction heads trained into the model; EAGLE-style methods use a small draft head on the model's hidden states. Speculation depth is how many tokens are drafted per step.

## What I took away

- No speculative decoding method wins everywhere; benchmark on your workload.
- Speculation depth is a key tuning knob.
- Read the vLLM blog for concrete enable and tune steps.

## Reference

- Original post: [https://x.com/vllm_project/status/2093148358143795254](https://x.com/vllm_project/status/2093148358143795254)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-08-28
