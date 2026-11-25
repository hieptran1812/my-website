---
title: "Deletion Tuning Teaches LLMs By Zeroing Weights"
publishDate: "2026-11-12"
date: "2026-11-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "ai-research", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2120850924969345352-1.jpg"
excerpt: "A paper claims zeroing under 0.1% of weights can match full fine-tuning while avoiding catastrophic forgetting."
---

> Source: [@qingke_ai on X](https://x.com/qingke_ai/status/2120850924969345352) · posted 2026-11-12

![Deletion Tuning Teaches LLMs By Zeroing Weights](/imgs/blogs/x-bookmarks/2120850924969345352-1.jpg)

## Summary

The commenter calls it counterintuitive that removing parameters can help learn new things better than adding them. The quoted post introduces **Deletion Tuning** (NeurIPS 2026): zeroing fewer than 0.1% of weights is claimed to match full fine-tuning accuracy at scale, be thousands of times smaller, and avoid forgetting.

## Background

Fine-tuning normally updates many weights, which can erase old skills (catastrophic forgetting). Here the edit is just setting a small set of weights to zero. If it works, the update is tiny to store and may suggest that fine-tuning mostly adjusts a few existing pathways. Results are the authors' claims.

## What I took away

- Tiny edits might carry fine-tuning gains, which helps continual learning.
- Smaller updates are cheaper to store and swap.
- Wait for replication before relying on the claim.

## Reference

- Original post: [https://x.com/qingke_ai/status/2120850924969345352](https://x.com/qingke_ai/status/2120850924969345352)
- Author: [@qingke_ai](https://x.com/qingke_ai) on X, posted 2026-11-12
