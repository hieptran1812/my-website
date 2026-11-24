---
title: "Effective Learning Rate Insights in LLM Training"
publishDate: "2026-08-30"
date: "2026-08-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A technical report using the MuonH optimizer discusses effective learning rate, echoing insights from a Tencent research blog."
---

> Source: [@vfleaking on X](https://x.com/vfleaking/status/2094105782501638358) · posted 2026-08-30

## Summary

The author thanks Zhanpeng for sharing their work and says the effective learning rate view is very useful for training LLMs. The quoted post links a detailed technical report that uses **MuonH** as its optimizer and discusses effective learning rate, with insights similar to a Tencent blog on ELR.

## Background

The learning rate sets how big each training update is. The effective learning rate is the actual size of the change relative to the weights, which can differ from the nominal setting because of weight norms, normalization and the optimizer. Muon is a newer optimizer for matrix weights; MuonH is a variant named in the post.

## What I took away

- The nominal learning rate can mislead; track the effective update size instead.
- Optimizer choice changes how learning rate behaves.
- Read the report and the ELR blog together for the full argument.

## Reference

- Original post: [https://x.com/vfleaking/status/2094105782501638358](https://x.com/vfleaking/status/2094105782501638358)
- Author: [@vfleaking](https://x.com/vfleaking) on X, posted 2026-08-30
