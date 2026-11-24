---
title: "Requential Coding Compresses LLMs Below One Bit Per Parameter"
publishDate: "2026-07-14"
date: "2026-07-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "ai-research", "math"]
author: "Hiep Tran"
featured: false
excerpt: "A paper introduces requential coding, claiming under 1-bit per parameter compression and an explanation for continued scaling gains."
---

> Source: [@Gabogonzalez515 on X](https://x.com/Gabogonzalez515/status/2077081882416882170) · posted 2026-07-14

## Summary

The quoted post introduces requential coding, which the authors say compresses billion-parameter LLMs to under 1 bit per parameter. They also say it explains why scaling does not hit a generalization wall. The author simply calls it a great paper.

## Background

Normal weights use 16 bits each; quantization shrinks that. Under 1 bit per parameter means the description is smaller than one binary choice per weight, which links to information theory ideas: a model that compresses well is simple in a deep sense, and simpler models tend to generalize.

## What I took away

- Compression and generalization are closely linked in theory.
- Information-theoretic compression bounds are different from practical runnable quantization.
- Read the paper to see how the scaling claim is supported.

## Reference

- Original post: [https://x.com/Gabogonzalez515/status/2077081882416882170](https://x.com/Gabogonzalez515/status/2077081882416882170)
- Author: [@Gabogonzalez515](https://x.com/Gabogonzalez515) on X, posted 2026-07-14
