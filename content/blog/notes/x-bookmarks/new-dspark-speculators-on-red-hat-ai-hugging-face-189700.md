---
title: "New DSpark Speculators on Red Hat AI Hugging Face"
publishDate: "2026-08-07"
date: "2026-08-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Red Hat AI released DSpark speculators for Qwen3.6-35B-A3B, gemma-4-31B and an updated GLM-5.2."
---

> Source: [@RedHat_AI on X](https://x.com/RedHat_AI/status/2085821099653189700) · posted 2026-08-07

## Summary

Red Hat AI released three DSpark speculators on Hugging Face for faster inference through parallel drafting: Qwen3.6-35B-A3B, gemma-4-31B and an updated GLM-5.2. DSpark drafts a whole block of tokens in one pass and then verifies losslessly. On GLM-5.2, they report 4.0 times faster decode.

## Background

A speculator is a draft model paired with a larger target model for speculative decoding. Drafting a block in parallel, instead of one token at a time, makes drafting itself faster. Lossless verification means the final output matches what the target model would produce alone.

## What I took away

- Check whether a ready-made speculator exists before training your own.
- Measure real speedup on your prompts; it depends on how predictable the text is.

## Reference

- Original post: [https://x.com/RedHat_AI/status/2085821099653189700](https://x.com/RedHat_AI/status/2085821099653189700)
- Author: [@RedHat_AI](https://x.com/RedHat_AI) on X, posted 2026-08-07
