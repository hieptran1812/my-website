---
title: "Learning Notes on Gated Delta Net and Linear Attention"
publishDate: "2026-08-06"
date: "2026-08-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "Day 5 of an inference engineering learning log covers Gated Delta Net and Kimi Delta Net attention papers."
---

> Source: [@not_ellington on X](https://x.com/not_ellington/status/2085194032335196398) · posted 2026-08-06

## Summary

On day 5 of an inference engineering log, the author spent the day reading the Gated Delta Net and Kimi Delta Net attention papers. He believes these linear memory architectures are the future and expects OpenAI and Anthropic to adopt them. The quoted day 4 post covered memory hardware: HBF, NAND flash, HBM, and how flash cells are tiered as SLC, MLC, TLC and QLC.

## Background

Linear attention variants such as delta nets keep a fixed-size state instead of a KV cache that grows with every token, so memory cost stays flat on long contexts. SLC to QLC describes how many bits each flash cell stores, trading speed and durability for density.

## What I took away

- Linear attention trades some exactness for constant memory on long contexts.
- Inference engineering spans both model architecture and memory hardware.
- Predictions about what big labs will adopt are the author's opinion.

## Reference

- Original post: [https://x.com/not_ellington/status/2085194032335196398](https://x.com/not_ellington/status/2085194032335196398)
- Author: [@not_ellington](https://x.com/not_ellington) on X, posted 2026-08-06
