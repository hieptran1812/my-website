---
title: "DiffusionBlocks Block-Wise Training Without End-to-End Backprop"
publishDate: "2026-05-27"
date: "2026-05-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2059648995132367277-1.jpg"
excerpt: "Sakana AI presents DiffusionBlocks, which trains neural networks in separate blocks to avoid holding the whole network in memory."
---

> Source: [@hardmaru on X](https://x.com/hardmaru/status/2059648995132367277) · posted 2026-05-27

![DiffusionBlocks Block-Wise Training Without End-to-End Backprop](/imgs/blogs/x-bookmarks/2059648995132367277-1.jpg)

## Summary

David Ha says that for over a decade end-to-end backprop has been the only way to train deep networks, but holding the whole network in memory causes a resource wall. They found a way to break the network into blocks and train them separately. The quote introduces DiffusionBlocks, block-wise training via a diffusion interpretation.

## Background

Backpropagation passes gradients through the full network, so all activations must be stored. Training blocks independently could cut memory use. The approach is a research claim and results are in the paper.

## What I took away

- Memory is a major limit of training, not only compute.
- Block-wise training is an alternative to end-to-end backprop.
- Check the paper for quality versus standard training.

## Reference

- Original post: [https://x.com/hardmaru/status/2059648995132367277](https://x.com/hardmaru/status/2059648995132367277)
- Author: [@hardmaru](https://x.com/hardmaru) on X, posted 2026-05-27
