---
title: "CoT Controllability Eval for Monitorable Reasoning"
publishDate: "2026-03-05"
date: "2026-03-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "interpretability"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2029625502089248878-1.jpg"
excerpt: "A new eval tests whether LLMs can control their chain of thought to evade monitors."
---

> Source: [@tomekkorbak on X](https://x.com/tomekkorbak/status/2029625502089248878) · posted 2026-03-05

![CoT Controllability Eval for Monitorable Reasoning](/imgs/blogs/x-bookmarks/2029625502089248878-1.jpg)

## Summary

The authors present CoT Controllability, an eval for whether LLMs can control their chains of thought to evade CoT monitors. They say results leave them cautiously optimistic because today's models struggle to obfuscate their reasoning.

## Background

Chain of thought (CoT) is the reasoning text a model writes before answering. Monitors read it to catch bad behavior. If models could hide their reasoning, monitoring would fail, so measuring this ability is a safety check.

## What I took away

- CoT monitoring depends on models not hiding reasoning.
- Current models struggle to obfuscate, per the authors.
- Re-run such evals as models improve.

## Reference

- Original post: [https://x.com/tomekkorbak/status/2029625502089248878](https://x.com/tomekkorbak/status/2029625502089248878)
- Author: [@tomekkorbak](https://x.com/tomekkorbak) on X, posted 2026-03-05
