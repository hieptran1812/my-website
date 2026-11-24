---
title: "Data Curation Should Depend on Available Compute"
publishDate: "2026-10-20"
date: "2026-10-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "learning-resources", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher highlights a paper arguing data selection should depend on compute budget, from a reading list on Marin's pretraining data."
---

> Source: [@cwolferesearch on X](https://x.com/cwolferesearch/status/2112373934419042804) · posted 2026-10-20

## Summary

@cwolferesearch was reading a thread about Marin's pretraining data and found a good list of data curation papers. He highlights one (arXiv 2404.07177) whose core argument is that data selection should depend on available compute. The quoted list includes Organize the Web, FineWeb-Edu, Data Curation cannot be Compute Agnostic, RegMix and OLMix.

## Background

Data curation means choosing and filtering training data. With little compute, training on a small high-quality subset may be best; with more compute, repeating that subset too often hurts, so broader data becomes useful. Data mixing papers like RegMix study how to weight different sources.

## What I took away

- The best data filter changes with your compute budget.
- Aggressive quality filtering can backfire when data gets repeated many times.
- Use the quoted list as a reading list on data curation.

## Reference

- Original post: [https://x.com/cwolferesearch/status/2112373934419042804](https://x.com/cwolferesearch/status/2112373934419042804)
- Author: [@cwolferesearch](https://x.com/cwolferesearch) on X, posted 2026-10-20
