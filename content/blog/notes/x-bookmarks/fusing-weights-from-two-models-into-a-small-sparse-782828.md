---
title: "Fusing Weights From Two Models Into a Small Sparse Model"
publishDate: "2026-08-06"
date: "2026-08-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "Maxime Labonne reacts to a sub-6B sparse model made by fusing weights from LFM2.5-2.6B and Qwen3.6-35B-A3B."
---

> Source: [@maximelabonne on X](https://x.com/maximelabonne/status/2085384100416782828) · posted 2026-08-06

## Summary

Maxime Labonne says the quoted project gives him flashbacks to early model merging and calls it wild in a positive way. In the quoted post, a developer releases a sub-6B sparse activation model built with a new architecture they call fusion, combining weights from Liquid AI's LFM2.5-2.6B and Qwen's Qwen3.6-35B-A3B. The developer claims near Qwen3.6-35B-A3B performance at a much smaller size.

## Background

Model merging combines the weights of separate trained models into one without full retraining. Sparse activation means only part of the network runs for each token, as in mixture-of-experts models, which keeps compute low.

## What I took away

- Model merging style experiments can produce surprising small models cheaply.
- Treat near big-model performance claims from a solo release as unverified until benchmarked.
- Combining models from different families is an open area for experimentation.

## Reference

- Original post: [https://x.com/maximelabonne/status/2085384100416782828](https://x.com/maximelabonne/status/2085384100416782828)
- Author: [@maximelabonne](https://x.com/maximelabonne) on X, posted 2026-08-06
