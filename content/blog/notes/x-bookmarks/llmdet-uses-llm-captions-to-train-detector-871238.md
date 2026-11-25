---
title: "LLMDet Uses LLM Captions To Train Detector"
publishDate: "2025-08-14"
date: "2025-08-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "ai-research", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1956065283224871238-1.jpg"
excerpt: "A post about LLMDet, which uses an LLM to annotate 1M images to train a Grounding DINO variant."
---

> Source: [@NielsRogge on X](https://x.com/NielsRogge/status/1956065283224871238) · posted 2025-08-14

![LLMDet Uses LLM Captions To Train Detector](/imgs/blogs/x-bookmarks/1956065283224871238-1.jpg)

## Summary

The post says LLMDet uses an LLM to densely annotate 1M images, and a Grounding DINO variant is trained on them. It links to models on Hugging Face and a zero-shot arena to compare against OWL-ViT and Grounding DINO.

## Background

Grounding DINO and OWL-ViT are open-vocabulary object detectors, which find objects described by text rather than a fixed label list. Dense annotations are detailed text descriptions of image contents. Using an LLM to write them is a way to scale training data.

## What I took away

- LLMs can generate detailed labels to train vision models.
- A zero-shot arena lets you compare detectors yourself.
- Open-vocabulary detection avoids fixed class lists.

## Reference

- Original post: [https://x.com/NielsRogge/status/1956065283224871238](https://x.com/NielsRogge/status/1956065283224871238)
- Author: [@NielsRogge](https://x.com/NielsRogge) on X, posted 2025-08-14
