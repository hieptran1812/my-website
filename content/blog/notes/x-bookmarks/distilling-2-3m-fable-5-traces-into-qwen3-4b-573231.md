---
title: "Distilling 2.3M Fable 5 Traces Into Qwen3-4B"
publishDate: "2026-07-03"
date: "2026-07-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2073171123542573231-1.jpg"
excerpt: "A post claims distilling 2.3M reasoning traces into Qwen3-4B gave perfect self-consistency and zero entropy."
---

> Source: [@waterloo_intern on X](https://x.com/waterloo_intern/status/2073171123542573231) · posted 2026-07-03

![Distilling 2.3M Fable 5 Traces Into Qwen3-4B (1)](/imgs/blogs/x-bookmarks/2073171123542573231-1.jpg)

![Distilling 2.3M Fable 5 Traces Into Qwen3-4B (2)](/imgs/blogs/x-bookmarks/2073171123542573231-2.jpg)

## Summary

@waterloo_intern claims they distilled 2.3M Claude Fable 5 reasoning traces into Qwen3-4B, with 100% self-consistency at 512 samples, 0.00 bits output entropy, zero hallucination variance, and that the student is not bounded by the teacher. They say they open-sourced it.

## Background

Self-consistency checks whether repeated samples agree. Zero entropy means identical outputs every time. Such results usually point to a measurement issue and should be verified.

## What I took away

- Extraordinary metrics need independent verification.
- Distillation can transfer a lot into small models.
- Another bookmark treats this as an export-control argument.

## Reference

- Original post: [https://x.com/waterloo_intern/status/2073171123542573231](https://x.com/waterloo_intern/status/2073171123542573231)
- Author: [@waterloo_intern](https://x.com/waterloo_intern) on X, posted 2026-07-03
