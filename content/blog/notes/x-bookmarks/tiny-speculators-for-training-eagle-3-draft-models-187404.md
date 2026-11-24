---
title: "Tiny-Speculators for Training EAGLE-3 Draft Models"
publishDate: "2026-07-20"
date: "2026-07-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "The author introduces tiny-speculators, a project to understand training an EAGLE-3 draft model for Qwen3-8B from scratch."
---

> Source: [@junupark_ on X](https://x.com/junupark_/status/2079216000592187404) · posted 2026-07-20

## Summary

The author introduces **tiny-speculators**, inspired by Speculators from Red Hat AI and the vLLM team. The goal was to understand the end-to-end process of training an EAGLE-3 draft model for Qwen3-8B from scratch. A link is provided.

## Background

EAGLE-3 is a speculative decoding method where a small draft head uses the main model's hidden features to predict upcoming tokens. Training such a drafter involves collecting the main model's features and teaching the drafter to match its predictions. Small from-scratch projects make this process easier to learn.

## What I took away

- Building a small version is a good way to learn speculative decoding.
- EAGLE-style drafters reuse the main model's hidden states.
- Use the project as a reference before training drafters for larger models.

## Reference

- Original post: [https://x.com/junupark_/status/2079216000592187404](https://x.com/junupark_/status/2079216000592187404)
- Author: [@junupark_](https://x.com/junupark_) on X, posted 2026-07-20
