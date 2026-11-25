---
title: "HY-WU Functional Neural Memory for Image Editing"
publishDate: "2026-03-05"
date: "2026-03-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Tencent Hunyuan shares the blog and code links for HY-WU, a framework generating LoRA updates for image editing."
---

> Source: [@TencentHunyuan on X](https://x.com/TencentHunyuan/status/2029644544195940775) · posted 2026-03-05

## Summary

This post gives blog and GitHub links for HY-WU, an extensible functional neural memory framework for text-guided image editing. The card says it generates low-rank (LoRA) updates on the fly, conditioned on the input.

## Background

LoRA is a small set of extra weights that adapts a model cheaply. Generating them per input means the model adapts for each request instead of staying fixed. This is the final post of a thread.

## What I took away

- Per-input generated LoRA is an alternative to a single static model.
- See the blog and repo for details.

## Reference

- Original post: [https://x.com/TencentHunyuan/status/2029644544195940775](https://x.com/TencentHunyuan/status/2029644544195940775)
- Author: [@TencentHunyuan](https://x.com/TencentHunyuan) on X, posted 2026-03-05
