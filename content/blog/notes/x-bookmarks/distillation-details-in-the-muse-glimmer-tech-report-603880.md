---
title: "Distillation Details in the Muse Glimmer Tech Report"
publishDate: "2026-10-19"
date: "2026-10-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "open-models", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher highlights the distillation recipe in the technical report for Muse Glimmer, a 30B open-weight agentic model."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2112276389210603880) · posted 2026-10-19

## Summary

@eliebakouch finds the report interesting, especially the distillation part: a full KL objective against an approximate teacher distribution, then full cross-entropy for the last 5% of training, and it is co-distillation. The quoted post introduces Muse Glimmer, a 30B open-weight model for agentic tasks that runs at practical speed on a Mac or PC with one consumer GPU at 4-bit, and claims it leads similar-size open models on tool use.

## Background

Distillation trains a smaller student model to imitate a larger teacher. A KL objective matches the student's full probability distribution to the teacher's, which carries more information than matching only the correct token (cross-entropy). Co-distillation means models learn from each other during training.

## What I took away

- Matching full teacher distributions gives a richer training signal than plain labels.
- Switching to cross-entropy near the end is a recipe detail worth testing.
- Small agentic models are being designed to run on consumer hardware.

## Reference

- Original post: [https://x.com/eliebakouch/status/2112276389210603880](https://x.com/eliebakouch/status/2112276389210603880)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-10-19
