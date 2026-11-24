---
title: "Dimuon Optimizer for LoRA Fine-Tuning"
publishDate: "2026-10-15"
date: "2026-10-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "Thinking Machines introduces Dimuon, a Muon variant for LoRA fine-tuning that often beats AdamW in their SFT tests."
---

> Source: [@cHHillee on X](https://x.com/cHHillee/status/2110569587955908774) · posted 2026-10-15

## Summary

@cHHillee says, quoting John Schulman, that there are many ways to spend compute to save customers compute, and this is one: a Muon variant that works much better for LoRA fine-tuning. The quoted Tinker post introduces Dimuon, which in their SFT tests often reaches lower loss in fewer steps than AdamW, and invites feedback.

## Background

LoRA fine-tunes a model by training small low-rank matrices instead of all weights, saving memory. Muon is an optimizer that orthogonalizes updates for weight matrices and has shown gains in pretraining. AdamW is the standard default optimizer.

## What I took away

- Optimizer choice can matter for LoRA, not only for pretraining.
- Lower loss in SFT does not always mean better downstream results; check benchmarks.
- See the eliebakouch note in this batch on optimizer mismatch risks.

## Reference

- Original post: [https://x.com/cHHillee/status/2110569587955908774](https://x.com/cHHillee/status/2110569587955908774)
- Author: [@cHHillee](https://x.com/cHHillee) on X, posted 2026-10-15
