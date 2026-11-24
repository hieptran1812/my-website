---
title: "On-Policy Self-Distillation With Better Hints"
publishDate: "2026-10-07"
date: "2026-10-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "A post arguing on-policy self-distillation works if hints are done well, citing Applied Compute's continual learning work."
---

> Source: [@benscharfstein on X](https://x.com/benscharfstein/status/2107909043176255882) · posted 2026-10-07

## Summary

The author mocks the claim that OPSD does not work, suggesting people may be doing it wrong. The quoted post from Applied Compute describes on-policy self-distillation, where a teacher model supervises training using privileged information in the form of a hint. It says the training signal depends heavily on hint quality, and manually refining hints is hard, which they aim to address as a step toward continual learning.

## Background

On-policy self-distillation means the model learns from its own generations, with a version of itself that sees extra information (a hint) acting as teacher. Continual learning means updating a model over time without forgetting old skills.

## What I took away

- Negative results can come from poor implementation, such as weak hints.
- If you try OPSD, invest in hint quality.
- Continual learning remains an open problem.

## Reference

- Original post: [https://x.com/benscharfstein/status/2107909043176255882](https://x.com/benscharfstein/status/2107909043176255882)
- Author: [@benscharfstein](https://x.com/benscharfstein) on X, posted 2026-10-07
