---
title: "Unsloth Guide To Fine-Tuning LLMs"
publishDate: "2025-11-21"
date: "2025-11-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "learning-resources", "quantization"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1991741860449251343-1.jpg"
excerpt: "A post pointing to Unsloth's guide on fine-tuning, from model selection and data to quantization, LoRA and evaluation."
---

> Source: [@TheGlobalMinima on X](https://x.com/TheGlobalMinima/status/1991741860449251343) · posted 2025-11-21

![Unsloth Guide To Fine-Tuning LLMs](/imgs/blogs/x-bookmarks/1991741860449251343-1.jpg)

## Summary

The author says Unsloth has a great fine-tuning guide covering model selection, data preparation, quantization and LoRA, training parameters and hyperparameters, and evaluation. They claim a small language model fine-tuned for a specific task often beats a generic LLM.

## Background

Fine-tuning adapts a pretrained model to your task with extra training. LoRA trains small add-on weights, and quantization shrinks the model so it fits smaller GPUs. Small tuned models can be cheaper to run for narrow tasks.

## What I took away

- Good data matters as much as the training settings.
- A small tuned model can be enough for a narrow task.
- Always evaluate against a baseline.

## Reference

- Original post: [https://x.com/TheGlobalMinima/status/1991741860449251343](https://x.com/TheGlobalMinima/status/1991741860449251343)
- Author: [@TheGlobalMinima](https://x.com/TheGlobalMinima) on X, posted 2025-11-21
