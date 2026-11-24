---
title: "Distilling a 4B Qwen Model to Say I Do Not Know"
publishDate: "2026-08-17"
date: "2026-08-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "ai-safety"]
author: "Hiep Tran"
featured: false
excerpt: "A blog on distillation to teach a small Qwen model calibrated uncertainty, with all artifacts released."
---

> Source: [@ritwik_raha on X](https://x.com/ritwik_raha/status/2089218793071649113) · posted 2026-08-17

## Summary

The author says they released all artifacts from their blog. The quoted post links a blog about using distillation to teach a 4B Qwen model to say I do not know, focusing on calibrated uncertainty in multiple-choice questions.

## Background

Distillation trains a smaller student model to imitate a larger teacher. Calibration means a model's confidence matches how often it is actually right; a calibrated model can admit uncertainty instead of guessing confidently. This reduces harmful hallucinations in practice.

## What I took away

- Teaching models to abstain can be more useful than squeezing out a little more accuracy.
- Released artifacts make it possible to reproduce and extend the work.
- Measure calibration, not only accuracy, when evaluating models.

## Reference

- Original post: [https://x.com/ritwik_raha/status/2089218793071649113](https://x.com/ritwik_raha/status/2089218793071649113)
- Author: [@ritwik_raha](https://x.com/ritwik_raha) on X, posted 2026-08-17
