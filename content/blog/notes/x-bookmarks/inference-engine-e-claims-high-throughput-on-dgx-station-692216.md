---
title: "Inference Engine e Claims High Throughput on DGX Station"
publishDate: "2026-10-15"
date: "2026-10-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "A company claims its new inference engine reaches 10,500 tok/s aggregate on one DGX Station with Nemotron 3.5 Super VL."
---

> Source: [@somewheresy on X](https://x.com/somewheresy/status/2110759944370692216) · posted 2026-10-15

## Summary

@somewheresy says they built a new inference engine called e, calls it the closest thing to alien technology, and claims 10,500 tokens per second aggregate on a single DGX Station with NVIDIA's Nemotron 3.5 Super VL. The quoted post links a case study on turning a DGX Station into a department-scale AI factory.

## Background

Aggregate throughput is total tokens per second across many simultaneous requests, which is much higher than single-user speed. A DGX Station is a desktop-class NVIDIA system for AI work. Throughput claims depend heavily on batch size and prompt lengths.

## What I took away

- Distinguish aggregate throughput from per-user speed when reading benchmarks.
- Ask for the batch size and sequence lengths behind a throughput claim.
- Marketing language like alien technology is not evidence.

## Reference

- Original post: [https://x.com/somewheresy/status/2110759944370692216](https://x.com/somewheresy/status/2110759944370692216)
- Author: [@somewheresy](https://x.com/somewheresy) on X, posted 2026-10-15
