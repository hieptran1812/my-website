---
title: "Predicting Hard Samples From Representation Angles"
publishDate: "2026-06-15"
date: "2026-06-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "interpretability", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2066554953951297867-1.png"
excerpt: "A post on a paper that finds difficult data samples by measuring angles between representations."
---

> Source: [@ruochenz_ on X](https://x.com/ruochenz_/status/2066554953951297867) · posted 2026-06-15

![Predicting Hard Samples From Representation Angles](/imgs/blogs/x-bookmarks/2066554953951297867-1.png)

## Summary

@ruochenz_ announces work showing that the angles between representations can pick out difficult data samples, useful for hard test sets or efficient training sets. The quoted post asks how to anticipate where an LLM will fail without evaluating specific examples.

## Background

Representations are the internal vectors a model builds for inputs. The idea is that geometry in this space reveals which inputs are hard. That could let developers find failure areas cheaply instead of running full evaluations.

## What I took away

- Geometry of representations can signal difficulty.
- Useful for building hard test sets.
- Read the thread for results and visuals.

## Reference

- Original post: [https://x.com/ruochenz_/status/2066554953951297867](https://x.com/ruochenz_/status/2066554953951297867)
- Author: [@ruochenz_](https://x.com/ruochenz_) on X, posted 2026-06-15
