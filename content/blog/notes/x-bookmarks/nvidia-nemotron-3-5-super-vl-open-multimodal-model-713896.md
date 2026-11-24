---
title: "NVIDIA Nemotron 3.5 Super VL Open Multimodal Model"
publishDate: "2026-10-15"
date: "2026-10-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "NVIDIA introduces Nemotron 3.5 Super VL, an open 120B-total, 12B-active model for long-running multimodal agents."
---

> Source: [@NVIDIAAI on X](https://x.com/NVIDIAAI/status/2110749836450713896) · posted 2026-10-15

## Summary

@NVIDIAAI announces Nemotron 3.5 Super VL, an open model that combines reasoning and tool use with text, image, document and video understanding for long-running AI agents. It has 120B total parameters with 12B active.

## Background

A model with 120B total and 12B active parameters is a mixture-of-experts: only a fraction of weights run per token, so it is cheaper to run than a dense 120B model. VL means vision-language, so it accepts images and video as well as text.

## What I took away

- Low active-parameter counts make large open models cheaper to serve.
- Agent models increasingly need to read documents and video, not just text.
- See the e inference engine note in this batch for a serving claim on this model.

## Reference

- Original post: [https://x.com/NVIDIAAI/status/2110749836450713896](https://x.com/NVIDIAAI/status/2110749836450713896)
- Author: [@NVIDIAAI](https://x.com/NVIDIAAI) on X, posted 2026-10-15
