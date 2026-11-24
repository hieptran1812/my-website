---
title: "Fine-Tuned Small Model Beats Frontier on Fraud Task"
publishDate: "2026-10-11"
date: "2026-10-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A post about Coinbase fine-tuning Qwen3.5-9B on internal data to beat frontier models on fraud prevention."
---

> Source: [@shampoo_capital on X](https://x.com/shampoo_capital/status/2109430987847536715) · posted 2026-10-11

## Summary

The author jokes that it is a cute intern project and hopes the intern got a return offer. The quoted post says Coinbase fine-tuned the open-weight Qwen3.5-9B on proprietary internal data and it outperformed frontier models on a specialized fraud-prevention task, at much lower cost. The quoted author expects to see more of this pattern.

## Background

Fine-tuning means further training a pretrained model on your own data. For narrow tasks with good private data, a small tuned model can beat a large general model while being cheaper and faster to run.

## What I took away

- For narrow, well-defined tasks, try fine-tuning a small open model before paying for frontier APIs.
- Proprietary data is often the real advantage, not model size.
- Compare on your own held-out set to confirm gains.

## Reference

- Original post: [https://x.com/shampoo_capital/status/2109430987847536715](https://x.com/shampoo_capital/status/2109430987847536715)
- Author: [@shampoo_capital](https://x.com/shampoo_capital) on X, posted 2026-10-11
