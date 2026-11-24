---
title: "Fine-Tuning Gemma 270M for On-Device Use"
publishDate: "2026-07-16"
date: "2026-07-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "quantization", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A summary of a talk showing a small Gemma model fine-tuned with synthetic data and LoRA, then quantized and run on a phone."
---

> Source: [@h100envy on X](https://x.com/h100envy/status/2077784077604692440) · posted 2026-07-16

## Summary

The author says a Google engineer explained how to fine-tune a tiny LLM from 46% to 90% accuracy and run it on a phone in 21 minutes. The recipe: pick Gemma 270M, generate synthetic task data, fine-tune with LoRA, quantize to int4, and deploy to a Pixel. The comparison to paid bootcamps is the author's framing.

## Background

LoRA fine-tunes a model by training small low-rank add-on matrices instead of all weights. Int4 quantization stores weights in 4 bits, shrinking memory so the model fits on a phone. Synthetic data means using a larger model to generate training examples for a narrow task.

## What I took away

- Tiny models can become strong on one narrow task with targeted fine-tuning.
- The pipeline synthetic data, LoRA, quantize, deploy is a reusable template.
- Measure accuracy after quantization, not only before.

## Reference

- Original post: [https://x.com/h100envy/status/2077784077604692440](https://x.com/h100envy/status/2077784077604692440)
- Author: [@h100envy](https://x.com/h100envy) on X, posted 2026-07-16
