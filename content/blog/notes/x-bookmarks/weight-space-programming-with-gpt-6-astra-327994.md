---
title: "Weight-Space Programming With GPT-6 Astra"
publishDate: "2026-11-02"
date: "2026-11-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "GPT-6 Astra edited 0.016% of Qwen3.5-0.8B weights to fix reasoning loops and boost math without training."
---

> Source: [@rikiyatakehi on X](https://x.com/rikiyatakehi/status/2117313742891327994) · posted 2026-11-02

## Summary

@rikiyatakehi reports asking GPT-6 Astra to edit Qwen3.5-0.8B weights to improve math. It changed only 0.016% of the weights, fixing reasoning loops and, they claim, improving math up to 6x and instruction following 3x, without gradients or training. They call this Weight-Space Programming.

## Background

Normally models improve through gradient-based training. Here another model edits weights directly. Other bookmarks reproduce the result, while critics say the gains largely come from fixing a known looping failure in thinking mode, and that standard interpretability tools did the work.

## What I took away

- Small targeted weight edits can change model behavior a lot.
- Large percentage gains on a broken baseline need careful reading.
- Follow the reproductions and critiques to get a balanced view.

## Reference

- Original post: [https://x.com/rikiyatakehi/status/2117313742891327994](https://x.com/rikiyatakehi/status/2117313742891327994)
- Author: [@rikiyatakehi](https://x.com/rikiyatakehi) on X, posted 2026-11-02
