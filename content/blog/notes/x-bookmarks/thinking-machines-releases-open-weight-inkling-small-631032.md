---
title: "Thinking Machines Releases Open-Weight Inkling-Small"
publishDate: "2026-07-30"
date: "2026-07-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "industry-news"]
author: "Hiep Tran"
featured: false
excerpt: "Thinking Machines releases Inkling-Small, a 276B total, 12B active parameter model said to match Inkling at a quarter of its size."
---

> Source: [@thinkymachines on X](https://x.com/thinkymachines/status/2082885869426631032) · posted 2026-07-30

## Summary

Thinking Machines announces **Inkling-Small**, which it says performs comparably to Inkling at a quarter of the size. It has 276B total parameters and 12B active. The full weights are available, and it can be fine-tuned on Tinker.

## Background

Total versus active parameters describes a mixture-of-experts model: all experts are stored, but only a few run per token, so compute cost follows the active count. Open weights let anyone download and fine-tune the model. Tinker is Thinking Machines' fine-tuning service.

## What I took away

- Active parameters drive inference cost; total parameters drive memory needs.
- Smaller models matching larger ones are claims to verify on your tasks.
- Open weights plus a fine-tuning service lower the barrier to customization.

## Reference

- Original post: [https://x.com/thinkymachines/status/2082885869426631032](https://x.com/thinkymachines/status/2082885869426631032)
- Author: [@thinkymachines](https://x.com/thinkymachines) on X, posted 2026-07-30
