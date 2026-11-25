---
title: "Why A Model Without Positional Encoding Still Works"
publishDate: "2026-02-18"
date: "2026-02-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "prajdabre asks what happens to a causal LM with no positional encodings."
---

> Source: [@prajdabre on X](https://x.com/prajdabre/status/2024095123722686763) · posted 2026-02-18

## Summary

The interview question says the answer is not garbage and asks why.

## Background

A causal mask lets each token attend only to earlier tokens, so a model can infer position from how many tokens it can see. This is known as NoPE.

## What I took away

- Causal masking leaks position.
- Positional encodings help extrapolation.

## Reference

- Original post: [https://x.com/prajdabre/status/2024095123722686763](https://x.com/prajdabre/status/2024095123722686763)
- Author: [@prajdabre](https://x.com/prajdabre) on X, posted 2026-02-18
